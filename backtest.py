import os
import time
import requests
import numpy as np
import pandas as pd
from tabulate import tabulate
from itertools import product

# =========================================================
# ONLY 4H TRADING + (OPTIONAL) HTF FILTER FROM NATIVE 1D DOWNLOAD
#
# - Baixa APENAS 4h para operar.
# - HTF só é permitido se vier de 1D BAIXADO SEPARADO (NÃO reamostrar do 4h).
#   Se não houver 1D nativo disponível, configs com HTF são removidas.
#
# SETUPS:
# - MA_LADDER:
#     * ARMA quando close>sma10 e sma10>sma15>...>sma45
#     * depois de armado, não precisa manter alinhado
#     * DESARMA se close < sma40
#     * "tocar na média" = low <= MA
#     * gatilho entry = high do candle + 1 tick (vale pro próximo candle)
#     * stop = abaixo da MA imediatamente abaixo da tocada (10->15, ..., 40->45, 45->45)
#     * enquanto não entra:
#         - ENTRY rola (high do candle atual + tick)
#         - STOP fica TRAVADO no valor do candle do toque mais baixo vigente
#         - se tocar MA mais baixa, atualiza o stop (novo valor) e trava de novo
#
# - PFR / DL / 8.2 / 8.3:
#     Só executam se "mercado direcional" no 4h:
#       * alinhamento: sma10>sma15>...>sma45
#       * sma10 apontando para cima (sma10 > sma10[-1])
#       * close> sma10 (TESTÁVEL com e sem)
#
# Experimentos A/B (configs):
# - HTF(native 1D) on/off (somente se 1D nativo existir)
# - Time-exit on/off
# - Break-even on/off (70% do alvo, efetivo no candle seguinte)
# - Require close>sma10 for classics/HTF (c10) on/off
#
# Ranking:
# - Split TREINO/TESTE
# - Ranking configs por Test AvgR no RR escolhido
# =========================================================

# =============================
# CONFIG (env-friendly)
# =============================
BASE_URL = os.getenv("MEXC_CONTRACT_BASE_URL", "https://contract.mexc.com/api/v1")
SYMBOL = os.getenv("SYMBOL", "BTC_USDT")
DEBUG = os.getenv("DEBUG", "0") == "1"

TRADING_TF = "4h"

SETUPS = [s.strip() for s in os.getenv("SETUPS", "MA_LADDER,PFR,DL,8.2,8.3").split(",") if s.strip()]
RANK_SETUP = os.getenv("RANK_SETUP", "ALL")  # "ALL" ou "MA_LADDER" etc.

# Ladder MAs
LADDER_MAS = [10, 15, 20, 25, 30, 35, 40, 45]
LADDER_DISARM_ON_CLOSE_BELOW = 40  # desarma se close < SMA40

# RR
RRS = [float(x) for x in os.getenv("RRS", "1.0,1.5,2.0").split(",")]
RANK_RR = float(os.getenv("RANK_RR", "1.5"))

# Download 4h
MAX_BARS_FETCH_4H = int(os.getenv("MAX_BARS_FETCH_4H", "12000"))
WINDOW_DAYS_4H = int(os.getenv("WINDOW_DAYS_4H", "90"))

# Download 1d (nativo) - usado só para HTF
DOWNLOAD_NATIVE_1D_FOR_HTF = os.getenv("DOWNLOAD_NATIVE_1D_FOR_HTF", "1") == "1"
MAX_BARS_FETCH_1D = int(os.getenv("MAX_BARS_FETCH_1D", "5000"))
WINDOW_DAYS_1D = int(os.getenv("WINDOW_DAYS_1D", "365"))

SAVE_OHLCV = os.getenv("SAVE_OHLCV", "1") == "1"

# HTF: exigir slope (sma10_up) no 1D
HTF_REQUIRE_SLOPE = os.getenv("HTF_REQUIRE_SLOPE", "1") == "1"

# Exits
TIME_EXIT_BARS = int(os.getenv("TIME_EXIT_BARS", "50"))              # em candles de 4h
BE_TARGET_FRACTION = float(os.getenv("BE_TARGET_FRACTION", "0.70"))  # 70% do caminho até o alvo

# Split treino/teste
TRAIN_FRACTION = float(os.getenv("TRAIN_FRACTION", "0.70"))
MIN_TRADES_FOR_RANK_TEST = int(os.getenv("MIN_TRADES_FOR_RANK_TEST", "80"))

# Segurança
MAX_CONFIGS = int(os.getenv("MAX_CONFIGS", "9999"))


# =============================
# HELPERS
# =============================
def rr_key(rr: float) -> str:
    if float(rr).is_integer():
        return str(int(rr))
    return str(rr)


def filter_for_rank(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if (RANK_SETUP or "").upper() == "ALL":
        return df
    return df[df["setup"] == RANK_SETUP].copy()


# =============================
# HTTP / DATA
# =============================
def http_get_json(url, params=None, tries=3, sleep_s=1.0):
    headers = {"User-Agent": "backtest-bot/1.0"}
    last_err = None
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            if r.status_code == 200:
                return r.json()
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(sleep_s)
    if DEBUG:
        print("http_get_json failed:", last_err)
    return None


def parse_kline(payload):
    if not payload or not isinstance(payload, dict):
        return pd.DataFrame()

    data = payload.get("data", None)
    if data is None:
        return pd.DataFrame()

    try:
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "vol"])
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

    df = df.rename(columns={
        "time": "ts",
        "timestamp": "ts",
        "t": "ts",
        "vol": "volume",
        "v": "volume",
        "amountVol": "volume",
    })

    required = ["ts", "open", "high", "low", "close", "volume"]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan
    df = df[required].copy()

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    if df.empty:
        return df

    if df["ts"].iloc[0] > 1e11:
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    else:
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    return df.sort_values("ts").reset_index(drop=True)


def fetch_history(symbol, interval_candidates, window_days, max_bars, label):
    print(f"Baixando histórico {label} para {symbol}...")

    end_ts = int(time.time())
    step = int(window_days) * 86400
    all_dfs = []
    total = 0

    for n in range(200):
        if total >= max_bars:
            break

        start_ts = end_ts - step

        df = pd.DataFrame()
        payload_last = None
        used_interval = None

        for interval in interval_candidates:
            url = f"{BASE_URL}/contract/kline/{symbol}"
            params = {"interval": interval, "start": start_ts, "end": end_ts}
            payload = http_get_json(url, params=params)
            payload_last = payload
            df = parse_kline(payload)
            if not df.empty:
                used_interval = interval
                break

        if DEBUG and n == 0:
            print(f"[{label}] candidates={interval_candidates} used={used_interval}")
            if isinstance(payload_last, dict):
                print(f"[{label}] payload keys:", list(payload_last.keys()))

        if df.empty:
            break

        all_dfs.append(df)
        total += len(df)

        first_ts_val = int(df.iloc[0]["ts"].timestamp())
        if first_ts_val >= end_ts:
            break
        end_ts = first_ts_val - 1

        time.sleep(0.2)

    if not all_dfs:
        return pd.DataFrame()

    full_df = (
        pd.concat(all_dfs)
        .drop_duplicates("ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )
    print(f"Total baixado: {len(full_df)} candles ({label}).")
    return full_df


def fetch_4h(symbol):
    return fetch_history(
        symbol=symbol,
        interval_candidates=["Hour4", "Min240"],
        window_days=WINDOW_DAYS_4H,
        max_bars=MAX_BARS_FETCH_4H,
        label="4h"
    )


def fetch_1d_native(symbol):
    return fetch_history(
        symbol=symbol,
        interval_candidates=["Day1", "Min1440"],
        window_days=WINDOW_DAYS_1D,
        max_bars=MAX_BARS_FETCH_1D,
        label="1d(native)"
    )


# =============================
# INDICATORS
# =============================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for p in LADDER_MAS:
        x[f"sma{p}"] = x["close"].rolling(p).mean()
    x["sma10_up"] = x["sma10"] > x["sma10"].shift(1)
    x["close_gt_sma10"] = x["close"] > x["sma10"]
    return x


# =============================
# MA alignment / regime helpers
# =============================
def ladder_aligned_strict(x, i) -> bool:
    for a, b in zip(LADDER_MAS[:-1], LADDER_MAS[1:]):
        va = x[f"sma{a}"].iloc[i]
        vb = x[f"sma{b}"].iloc[i]
        if np.isnan(va) or np.isnan(vb):
            return False
        if not (va > vb):
            return False
    return True


def classic_allowed_regime(x, i, cfg) -> bool:
    """
    Clássicos (PFR/DL/8.2/8.3) só rodam se:
      - alinhamento strict
      - sma10_up
      - e opcionalmente close>sma10 (cfg['req_close_gt10'])
    """
    if not ladder_aligned_strict(x, i):
        return False
    if not bool(x["sma10_up"].iloc[i]):
        return False
    if cfg["req_close_gt10"] and (not bool(x["close_gt_sma10"].iloc[i])):
        return False
    return True


# =============================
# SETUPS (clássicos)
# =============================
def check_signals_classic(x, i):
    pfr = (
        (x["low"].iloc[i] < x["low"].iloc[i - 1])
        and (x["low"].iloc[i] < x["low"].iloc[i - 2])
        and (x["close"].iloc[i] > x["close"].iloc[i - 1])
    )
    dl = (x["low"].iloc[i] < x["low"].iloc[i - 1]) and (x["low"].iloc[i] < x["low"].iloc[i - 2])
    s82 = x["close"].iloc[i] < x["low"].iloc[i - 1]
    s83 = (
        (x["close"].iloc[i] < x["close"].iloc[i - 2])
        and (x["close"].iloc[i - 1] < x["close"].iloc[i - 2])
        and (x["close"].iloc[i - 1] >= x["low"].iloc[i - 2])
    )
    return pfr, dl, s82, s83


# =============================
# MA_LADDER helpers
# =============================
def ladder_armed_condition(x, i) -> bool:
    """
    Arma quando close>sma10 e alinhamento strict.
    (Esse close>sma10 do LADDER fica fixo; não entra no A/B dos clássicos.)
    """
    if not ladder_aligned_strict(x, i):
        return False
    c = x["close"].iloc[i]
    sma10 = x["sma10"].iloc[i]
    if np.isnan(c) or np.isnan(sma10):
        return False
    return c > sma10


def ladder_disarm_condition(x, i) -> bool:
    sma40 = x["sma40"].iloc[i]
    c = x["close"].iloc[i]
    if np.isnan(sma40) or np.isnan(c):
        return False
    return c < sma40


def candle_touches_ma(low, ma_value) -> bool:
    # tocar = low <= MA
    if np.isnan(ma_value):
        return False
    return low <= ma_value


def deepest_touched_ladder_ma(x, i):
    lo = x["low"].iloc[i]
    touched = []
    for p in LADDER_MAS:
        v = x[f"sma{p}"].iloc[i]
        if candle_touches_ma(lo, v):
            touched.append(p)
    if not touched:
        return None
    return max(touched)


def ladder_below_period(touched_period: int) -> int:
    if touched_period not in LADDER_MAS:
        return LADDER_MAS[-1]
    idx = LADDER_MAS.index(touched_period)
    if idx == len(LADDER_MAS) - 1:
        return touched_period
    return LADDER_MAS[idx + 1]


# =============================
# TRADE SIM (per RR)
# =============================
def simulate_trade_per_rr(x, fill_idx, entry, initial_stop, rr, cfg):
    risk = entry - initial_stop
    if risk <= 0 or np.isnan(risk):
        return False, "invalid", np.nan, pd.NaT

    stop = initial_stop
    target = entry + (risk * rr)

    be_trigger = entry + (target - entry) * BE_TARGET_FRACTION
    be_pending = False
    be_active = False

    n_bars = max(int(TIME_EXIT_BARS), 1)
    last_idx = min(fill_idx + n_bars - 1, len(x) - 1)

    for k in range(fill_idx, last_idx + 1):
        c = x.iloc[k]

        if cfg["use_breakeven"] and be_pending and not be_active:
            stop = entry
            be_active = True
            be_pending = False

        if c["low"] <= stop:
            r = (stop - entry) / risk
            return False, "stop" if stop < entry else "breakeven_stop", float(r), c["ts"]

        if c["high"] >= target:
            return True, "target", float(rr), c["ts"]

        if cfg["use_breakeven"] and (not be_active) and (c["high"] >= be_trigger):
            be_pending = True

    if cfg["use_time_exit"]:
        exit_px = float(x["close"].iloc[last_idx])
        r = (exit_px - entry) / risk
        return False, "time_exit", float(r), x["ts"].iloc[last_idx]

    return False, "timeout_loss", -1.0, x["ts"].iloc[last_idx]


# =============================
# BACKTEST CORE (gera trades brutos)
# =============================
def run_backtest_generate_trades_4h(df_4h, cfg, htf_flags_1d_native=None):
    x = add_indicators(df_4h).copy()
    if x.empty:
        return pd.DataFrame()
    x = x.sort_values("ts").reset_index(drop=True)

    # Merge HTF 1D nativo
    if cfg["use_htf"]:
        if htf_flags_1d_native is None or htf_flags_1d_native.empty:
            return pd.DataFrame()

        tmp = pd.merge_asof(
            x[["ts"]],
            htf_flags_1d_native.sort_values("ts"),
            on="ts",
            direction="backward"
        )

        x["htf_aligned"] = tmp["htf_aligned"].fillna(False).astype(bool)
        x["htf_sma10_up"] = tmp["htf_sma10_up"].fillna(False).astype(bool)
        x["htf_close_gt10"] = tmp["htf_close_gt10"].fillna(False).astype(bool)
    else:
        x["htf_aligned"] = True
        x["htf_sma10_up"] = True
        x["htf_close_gt10"] = True

    def htf_ok(i) -> bool:
        if not cfg["use_htf"]:
            return True
        if not x["htf_aligned"].iloc[i]:
            return False
        if cfg["req_close_gt10"] and (not x["htf_close_gt10"].iloc[i]):
            return False
        if HTF_REQUIRE_SLOPE and (not x["htf_sma10_up"].iloc[i]):
            return False
        return True

    tick = float(x["close"].iloc[-1] * 0.0001)

    # precisa de SMA45 e de lookbacks de 2 candles pros clássicos
    start_idx = max(max(LADDER_MAS) + 5, 5)

    trades = []

    # ---------- Estado do MA_LADDER ----------
    ladder_enabled = "MA_LADDER" in SETUPS
    ladder_armed = False
    ladder_armed_ts = pd.NaT

    ladder_touched_period = None
    ladder_touch_idx = None
    ladder_stop_fixed = None  # STOP travado (valor fixo do candle do toque mais baixo)

    ladder_pending = False
    ladder_pending_entry = None
    ladder_pending_stop = None
    ladder_pending_signal_idx = None
    ladder_pending_touch_period = None
    ladder_pending_touch_idx = None

    for i in range(start_idx, len(x)):
        # 1) Fill ordem pendente do ladder
        if ladder_enabled and ladder_pending:
            if x["high"].iloc[i] >= ladder_pending_entry:
                fill_idx = i
                entry = float(ladder_pending_entry)
                stop = float(ladder_pending_stop)

                si = int(ladder_pending_signal_idx) if ladder_pending_signal_idx is not None else i
                si = max(0, min(si, len(x) - 1))

                ti = int(ladder_pending_touch_idx) if ladder_pending_touch_idx is not None else si
                ti = max(0, min(ti, len(x) - 1))

                row = {
                    "config": cfg["name"],
                    "timeframe": "4h",
                    "setup": "MA_LADDER",

                    "armed_ts": ladder_armed_ts,
                    "ladder_touch_ts": x["ts"].iloc[ti],
                    "signal_ts": x["ts"].iloc[si],
                    "fill_ts": x["ts"].iloc[fill_idx],

                    "entry": entry,
                    "stop": stop,
                    "risk": entry - stop,

                    "ladder_touch_ma": int(ladder_pending_touch_period) if ladder_pending_touch_period is not None else np.nan,
                    "ladder_stop_ma": int(ladder_below_period(ladder_pending_touch_period)) if ladder_pending_touch_period is not None else np.nan,

                    "use_htf": cfg["use_htf"],
                    "use_time_exit": cfg["use_time_exit"],
                    "use_breakeven": cfg["use_breakeven"],
                    "req_close_gt10": cfg["req_close_gt10"],
                }

                for rr in RRS:
                    hit, exit_type, r_res, exit_ts = simulate_trade_per_rr(
                        x=x, fill_idx=fill_idx, entry=entry, initial_stop=stop, rr=rr, cfg=cfg
                    )
                    k = rr_key(rr)
                    row[f"hit_{k}"] = bool(hit)
                    row[f"exit_type_{k}"] = exit_type
                    row[f"R_{k}"] = float(r_res) if r_res is not None else np.nan
                    row[f"exit_ts_{k}"] = exit_ts

                trades.append(row)

                # reset após entrada
                ladder_armed = False
                ladder_armed_ts = pd.NaT
                ladder_touched_period = None
                ladder_touch_idx = None
                ladder_stop_fixed = None

                ladder_pending = False
                ladder_pending_entry = None
                ladder_pending_stop = None
                ladder_pending_signal_idx = None
                ladder_pending_touch_period = None
                ladder_pending_touch_idx = None

        # 2) setups clássicos (com filtro direcional + HTF)
        if any(s in SETUPS for s in ["PFR", "DL", "8.2", "8.3"]):
            if i >= 2 and classic_allowed_regime(x, i, cfg) and htf_ok(i):
                pfr, dl, s82, s83 = check_signals_classic(x, i)

                active = []
                if pfr and "PFR" in SETUPS:
                    active.append("PFR")
                if dl and (not pfr) and "DL" in SETUPS:
                    active.append("DL")
                if s82 and "8.2" in SETUPS:
                    active.append("8.2")
                if s83 and "8.3" in SETUPS:
                    active.append("8.3")

                for setup in active:
                    max_wait = 3 if setup in ["8.2", "8.3"] else 1

                    entry = float(x["high"].iloc[i] + tick)
                    stop = float(x["low"].iloc[i] - tick)

                    filled = False
                    fill_idx = -1

                    for w in range(1, max_wait + 1):
                        if i + w >= len(x):
                            break
                        curr = x.iloc[i + w]

                        if curr["high"] >= entry:
                            filled = True
                            fill_idx = i + w
                            break

                        if setup in ["8.2", "8.3"]:
                            # rolagem do gatilho (mantida)
                            if curr["high"] < entry - tick:
                                entry = float(curr["high"] + tick)
                                stop = float(min(stop, curr["low"] - tick))
                        else:
                            break

                    if not filled:
                        continue

                    risk = entry - stop
                    if risk <= 0 or np.isnan(risk):
                        continue

                    row = {
                        "config": cfg["name"],
                        "timeframe": "4h",
                        "setup": setup,

                        "armed_ts": pd.NaT,
                        "ladder_touch_ts": pd.NaT,
                        "signal_ts": x["ts"].iloc[i],
                        "fill_ts": x["ts"].iloc[fill_idx],

                        "entry": entry,
                        "stop": stop,
                        "risk": risk,

                        "ladder_touch_ma": np.nan,
                        "ladder_stop_ma": np.nan,

                        "use_htf": cfg["use_htf"],
                        "use_time_exit": cfg["use_time_exit"],
                        "use_breakeven": cfg["use_breakeven"],
                        "req_close_gt10": cfg["req_close_gt10"],
                    }

                    for rr in RRS:
                        hit, exit_type, r_res, exit_ts = simulate_trade_per_rr(
                            x=x, fill_idx=fill_idx, entry=entry, initial_stop=stop, rr=rr, cfg=cfg
                        )
                        k = rr_key(rr)
                        row[f"hit_{k}"] = bool(hit)
                        row[f"exit_type_{k}"] = exit_type
                        row[f"R_{k}"] = float(r_res) if r_res is not None else np.nan
                        row[f"exit_ts_{k}"] = exit_ts

                    trades.append(row)

        # 3) MA_LADDER state machine
        if ladder_enabled:
            # desarma se close < sma40
            if ladder_armed and ladder_disarm_condition(x, i):
                ladder_armed = False
                ladder_armed_ts = pd.NaT
                ladder_touched_period = None
                ladder_touch_idx = None
                ladder_stop_fixed = None

                ladder_pending = False
                ladder_pending_entry = None
                ladder_pending_stop = None
                ladder_pending_signal_idx = None
                ladder_pending_touch_period = None
                ladder_pending_touch_idx = None
                continue

            # arma (com HTF ok, se HTF estiver ligado)
            if (not ladder_armed) and ladder_armed_condition(x, i) and htf_ok(i):
                ladder_armed = True
                ladder_armed_ts = x["ts"].iloc[i]
                ladder_touched_period = None
                ladder_touch_idx = None
                ladder_stop_fixed = None

                ladder_pending = False
                ladder_pending_entry = None
                ladder_pending_stop = None
                ladder_pending_signal_idx = None
                ladder_pending_touch_period = None
                ladder_pending_touch_idx = None

            if ladder_armed:
                # se HTF falhar, não rola ordem (mas mantém armado)
                if not htf_ok(i):
                    ladder_pending = False
                    ladder_pending_entry = None
                    ladder_pending_stop = None
                    ladder_pending_signal_idx = None
                    ladder_pending_touch_period = None
                    ladder_pending_touch_idx = None
                else:
                    touched = deepest_touched_ladder_ma(x, i)

                    # atualiza se tocou MA mais baixa
                    if touched is not None:
                        if ladder_touched_period is None or touched > ladder_touched_period:
                            ladder_touched_period = touched
                            ladder_touch_idx = i

                            below_p = ladder_below_period(ladder_touched_period)
                            below_ma_val = x[f"sma{below_p}"].iloc[i]
                            if np.isnan(below_ma_val):
                                ladder_stop_fixed = None
                            else:
                                ladder_stop_fixed = float(below_ma_val - tick)  # FIXO

                    if ladder_touched_period is not None and ladder_stop_fixed is not None:
                        ladder_pending = True
                        ladder_pending_entry = float(x["high"].iloc[i] + tick)  # ENTRY rola
                        ladder_pending_stop = float(ladder_stop_fixed)          # STOP travado
                        ladder_pending_signal_idx = i
                        ladder_pending_touch_period = ladder_touched_period
                        ladder_pending_touch_idx = ladder_touch_idx
                    else:
                        ladder_pending = False
                        ladder_pending_entry = None
                        ladder_pending_stop = None
                        ladder_pending_signal_idx = None
                        ladder_pending_touch_period = None
                        ladder_pending_touch_idx = None

    return pd.DataFrame(trades)


# =============================
# CONFIGS (variáveis estruturais)
# =============================
def build_structural_configs(htf_available: bool):
    configs = []
    use_htf_options = [False, True] if htf_available else [False]

    for use_htf, use_time_exit, use_breakeven, req_close_gt10 in product(
        use_htf_options, [False, True], [False, True], [False, True]
    ):
        name = f"htf{int(use_htf)}_tx{int(use_time_exit)}_be{int(use_breakeven)}_c10{int(req_close_gt10)}"
        configs.append({
            "name": name,
            "use_htf": use_htf,
            "use_time_exit": use_time_exit,
            "use_breakeven": use_breakeven,
            "req_close_gt10": req_close_gt10,
        })

    if len(configs) > MAX_CONFIGS:
        configs = configs[:MAX_CONFIGS]
    return configs


# =============================
# METRICS / SUMMARIES
# =============================
def split_train_test(trades_df):
    if trades_df.empty:
        return trades_df.copy(), trades_df.copy(), pd.NaT
    trades_df = trades_df.sort_values("fill_ts").reset_index(drop=True)
    cutoff = trades_df["fill_ts"].quantile(TRAIN_FRACTION)
    train = trades_df[trades_df["fill_ts"] <= cutoff].copy()
    test = trades_df[trades_df["fill_ts"] > cutoff].copy()
    return train, test, cutoff


def metrics_overall(trades_df, rr):
    if trades_df.empty:
        return {"Trades": 0, "WR": np.nan, "AvgR": np.nan}
    k = rr_key(rr)
    trades = len(trades_df)
    wr = float(trades_df[f"hit_{k}"].mean()) if trades > 0 else np.nan
    avgr = float(trades_df[f"R_{k}"].mean()) if trades > 0 else np.nan
    return {"Trades": trades, "WR": wr, "AvgR": avgr}


# =============================
# MAIN
# =============================
def main():
    os.makedirs("results", exist_ok=True)

    # 1) Baixa 4h
    df_4h = fetch_4h(SYMBOL)
    if df_4h.empty:
        print("Erro: Sem dados 4h baixados.")
        pd.DataFrame({"status": ["no_4h_data"]}).to_csv(f"results/only4h_{SYMBOL}.csv", index=False)
        return

    if SAVE_OHLCV:
        df_4h.to_csv(f"results/ohlcv_{SYMBOL}_4h.csv", index=False)

    # 2) Baixa 1d nativo (somente para HTF)
    htf_flags_1d_native = None
    htf_available = False

    if DOWNLOAD_NATIVE_1D_FOR_HTF:
        df_1d_native = fetch_1d_native(SYMBOL)
        if not df_1d_native.empty:
            if SAVE_OHLCV:
                df_1d_native.to_csv(f"results/ohlcv_{SYMBOL}_1d_native.csv", index=False)

            ind_1d = add_indicators(df_1d_native).sort_values("ts").reset_index(drop=True)

            # HTF flags: alinhamento + sma10_up + close>10 (esse último será opcional por config)
            aligned_list = []
            for i in range(len(ind_1d)):
                aligned_list.append(ladder_aligned_strict(ind_1d, i))

            ind_1d["aligned"] = pd.Series(aligned_list, index=ind_1d.index).astype(bool)

            htf_flags_1d_native = ind_1d[["ts", "aligned", "sma10_up", "close_gt_sma10"]].rename(columns={
                "aligned": "htf_aligned",
                "sma10_up": "htf_sma10_up",
                "close_gt_sma10": "htf_close_gt10",
            })

            htf_available = True
        else:
            print("Aviso: não foi possível baixar 1d nativo; configs com HTF serão DESABILITADAS.")
            htf_available = False
    else:
        print("DOWNLOAD_NATIVE_1D_FOR_HTF=0 => HTF desabilitado por regra.")
        htf_available = False

    # 3) Configs estruturais (inclui c10 0/1)
    configs = build_structural_configs(htf_available=htf_available)
    print(f"ONLY 4H | HTF(native1d) available={htf_available} | configs={len(configs)}")
    print(f"SETUPS={SETUPS} | RANK_SETUP={RANK_SETUP} | RANK_RR={RANK_RR}")
    print("Obs: configs incluem _c10{0|1} para testar close>sma10 nos clássicos/HTF.")

    # 4) Gera trades
    all_trades = []
    for cfg in configs:
        print(f"Gerando trades 4h: {cfg['name']} ...")
        try:
            t = run_backtest_generate_trades_4h(
                df_4h=df_4h,
                cfg=cfg,
                htf_flags_1d_native=htf_flags_1d_native
            )
            if not t.empty:
                all_trades.append(t)
        except Exception as e:
            print(f"Erro em {cfg['name']}: {e}")

    if not all_trades:
        print("0 trades gerados.")
        pd.DataFrame({"status": ["no_trades"]}).to_csv(f"results/only4h_{SYMBOL}.csv", index=False)
        return

    trades = pd.concat(all_trades, ignore_index=True)
    trades.to_csv(f"results/only4h_trades_raw_{SYMBOL}.csv", index=False)

    # 5) Treino/teste
    train, test, cutoff = split_train_test(trades)
    print(f"Cutoff treino/teste: {cutoff}")

    train_rank = filter_for_rank(train)
    test_rank = filter_for_rank(test)

    # 6) Stage 1 ranking
    stage1_rows = []
    for cfg_name, _ in trades.groupby("config"):
        tr = train_rank[train_rank["config"] == cfg_name]
        te = test_rank[test_rank["config"] == cfg_name]
        m_tr = metrics_overall(tr, RANK_RR)
        m_te = metrics_overall(te, RANK_RR)

        stage1_rows.append({
            "Config": cfg_name,
            "Train Trades": m_tr["Trades"],
            "Train WR": m_tr["WR"],
            "Train AvgR": m_tr["AvgR"],
            "Test Trades": m_te["Trades"],
            "Test WR": m_te["WR"],
            "Test AvgR": m_te["AvgR"],
            "Eligible": (m_te["Trades"] >= MIN_TRADES_FOR_RANK_TEST),
        })

    stage1 = pd.DataFrame(stage1_rows).sort_values(
        ["Eligible", "Test AvgR", "Test Trades"],
        ascending=[False, False, False]
    )
    stage1.to_csv(f"results/only4h_stage1_structural_{SYMBOL}.csv", index=False)

    print("\n=== Stage 1 (4h only) - ranking por Test AvgR @ RR="
          f"{RANK_RR} | RankSetup={RANK_SETUP} (min test trades={MIN_TRADES_FOR_RANK_TEST}) ===")
    print(tabulate(stage1, headers="keys", tablefmt="grid", showindex=False))

    with open(f"results/only4h_summary_{SYMBOL}.md", "w", encoding="utf-8") as f:
        f.write(f"# Only 4H Summary - {SYMBOL}\n\n")
        f.write("- Trades: somente 4h\n")
        f.write(f"- HTF: somente se 1D nativo baixar separado (available={htf_available})\n")
        f.write(f"- SETUPS={SETUPS}\n")
        f.write(f"- RANK_SETUP={RANK_SETUP} | RANK_RR={RANK_RR}\n")
        f.write(f"- Train fraction: {TRAIN_FRACTION}\n")
        f.write(f"- Time exit bars (4h): {TIME_EXIT_BARS}\n")
        f.write(f"- Break-even fraction of target: {BE_TARGET_FRACTION}\n")
        f.write(f"- HTF require slope: {HTF_REQUIRE_SLOPE}\n")
        f.write(f"- Note: configs have _c10{{0|1}} for close>sma10 gating on classics/HTF.\n\n")
        f.write("## Stage 1 (Structural)\n\n")
        f.write(tabulate(stage1, headers="keys", tablefmt="pipe", showindex=False))


if __name__ == "__main__":
    main()
