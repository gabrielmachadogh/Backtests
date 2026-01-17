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
# Regras do usuário:
# - Trades APENAS no 4h.
# - HTF só é permitido se vier de 1D BAIXADO SEPARADO (NÃO reamostrar do 4h).
#   Se não houver 1D nativo disponível, configs com HTF são removidas.
#
# Testes (variáveis ON/OFF):
# - HTF (1D nativo) on/off
# - Time-exit on/off (sai a mercado após N candles, calcula R real)
# - Break-even on/off (aciona ao atingir 70% do alvo; efetivo no candle seguinte)
#
# Stretch sweep (aplicado como filtro em cima dos trades já gerados):
# - dist_sma8_max, dist_sma80_max, sma_gap_min
#
# Validação "mais conclusiva":
# - Split temporal TREINO/TESTE (global)
# - Stage 1: escolhe melhor config estrutural (sem stretch)
# - Stage 2: escolhe melhor stretch no TREINO e valida no TESTE
# =========================================================

# =============================
# CONFIG (env-friendly)
# =============================
BASE_URL = os.getenv("MEXC_CONTRACT_BASE_URL", "https://contract.mexc.com/api/v1")
SYMBOL = os.getenv("SYMBOL", "BTC_USDT")
DEBUG = os.getenv("DEBUG", "0") == "1"

TRADING_TF = "4h"

SMA_SHORT = int(os.getenv("SMA_SHORT", "8"))
SMA_LONG = int(os.getenv("SMA_LONG", "80"))
SLOPE_LOOKBACK = int(os.getenv("SLOPE_LOOKBACK", "8"))

RRS = [float(x) for x in os.getenv("RRS", "1.0,1.5,2.0").split(",")]
RANK_RR = float(os.getenv("RANK_RR", "1.5"))  # RR usado para ranking

# Download 4h
MAX_BARS_FETCH_4H = int(os.getenv("MAX_BARS_FETCH_4H", "12000"))
WINDOW_DAYS_4H = int(os.getenv("WINDOW_DAYS_4H", "90"))

# Download 1d (nativo) - usado só para HTF
DOWNLOAD_NATIVE_1D_FOR_HTF = os.getenv("DOWNLOAD_NATIVE_1D_FOR_HTF", "1") == "1"
MAX_BARS_FETCH_1D = int(os.getenv("MAX_BARS_FETCH_1D", "5000"))
WINDOW_DAYS_1D = int(os.getenv("WINDOW_DAYS_1D", "365"))

SAVE_OHLCV = os.getenv("SAVE_OHLCV", "1") == "1"

# ----- HTF -----
HTF_REQUIRE_SLOPE = os.getenv("HTF_REQUIRE_SLOPE", "1") == "1"

# ----- Exits -----
TIME_EXIT_BARS = int(os.getenv("TIME_EXIT_BARS", "50"))            # em candles de 4h
BE_TARGET_FRACTION = float(os.getenv("BE_TARGET_FRACTION", "0.70"))  # 70% do caminho até o alvo

# ----- Stretch sweep (em % do SMA) -----
# dist_sma8  = (close - sma8) / sma8
# dist_sma80 = (close - sma80) / sma80
# sma_gap    = (sma8 - sma80) / sma80
DIST_SMA8_MAX_LIST = [float(x) for x in os.getenv("DIST_SMA8_MAX_LIST", "0.01,0.02,0.03").split(",")]
DIST_SMA80_MAX_LIST = [float(x) for x in os.getenv("DIST_SMA80_MAX_LIST", "0.02,0.03,0.05").split(",")]
SMA_GAP_MIN_LIST = [float(x) for x in os.getenv("SMA_GAP_MIN_LIST", "0.00,0.005,0.01").split(",")]

# Stage selection
TOP_K_CONFIGS_FOR_SWEEP = int(os.getenv("TOP_K_CONFIGS_FOR_SWEEP", "2"))
TRAIN_FRACTION = float(os.getenv("TRAIN_FRACTION", "0.70"))

# Mínimos para reduzir overfit por amostra pequena
MIN_TRADES_FOR_RANK_TEST = int(os.getenv("MIN_TRADES_FOR_RANK_TEST", "80"))
MIN_TRADES_FOR_SELECT_TRAIN = int(os.getenv("MIN_TRADES_FOR_SELECT_TRAIN", "80"))

# Segurança
MAX_CONFIGS = int(os.getenv("MAX_CONFIGS", "9999"))


# =============================
# HELPERS
# =============================
def rr_key(rr: float) -> str:
    if float(rr).is_integer():
        return str(int(rr))
    return str(rr)


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
    """
    Baixa candles usando uma lista de possíveis 'interval' (MEXC às vezes muda).
    """
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
    # mais comum: Hour4 ou Min240
    return fetch_history(
        symbol=symbol,
        interval_candidates=["Hour4", "Min240"],
        window_days=WINDOW_DAYS_4H,
        max_bars=MAX_BARS_FETCH_4H,
        label="4h"
    )


def fetch_1d_native(symbol):
    # em alguns ambientes: Day1 / Min1440
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
def add_indicators(df):
    x = df.copy()
    x["sma8"] = x["close"].rolling(SMA_SHORT).mean()
    x["sma80"] = x["close"].rolling(SMA_LONG).mean()

    # regime long-only (como você já vinha usando)
    x["trend_up"] = (x["close"] > x["sma8"]) & (x["close"] > x["sma80"]) & (x["sma8"] > x["sma80"])
    x["slope_up"] = x["sma8"] > x["sma8"].shift(1).rolling(SLOPE_LOOKBACK).max()

    # Stretch features (% do SMA)
    x["dist_sma8"] = (x["close"] - x["sma8"]) / x["sma8"]
    x["dist_sma80"] = (x["close"] - x["sma80"]) / x["sma80"]
    x["sma_gap"] = (x["sma8"] - x["sma80"]) / x["sma80"]

    return x


# =============================
# SETUPS
# =============================
def check_signals(x, i):
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
# TRADE SIM (per RR)
# =============================
def simulate_trade_per_rr(x, fill_idx, entry, initial_stop, rr, cfg):
    """
    Break-even:
      - dispara quando atingir 70% do alvo daquele RR
      - aplica somente a partir do candle seguinte (evita ambiguidade intrabar)
    Time-exit:
      - se não bater stop/target em N candles, sai a mercado (close) e calcula R real
    """
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

        # BE efetivo no início do candle seguinte ao trigger
        if cfg["use_breakeven"] and be_pending and not be_active:
            stop = entry
            be_active = True
            be_pending = False

        # Ordem conservadora intrabar:
        # 1) stop
        # 2) target
        # 3) trigger BE (para vigorar só no próximo candle)
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
# BACKTEST CORE (gera trades "brutos" - sem stretch filter)
# =============================
def run_backtest_generate_trades_4h(df_4h, cfg, htf_flags_1d_native=None):
    """
    Roda tudo no 4h.
    HTF (opcional) usa SOMENTE flags de 1D NATIVO (se disponível).
    """
    x = add_indicators(df_4h).copy()
    if x.empty or len(x) < (SMA_LONG + 30):
        return pd.DataFrame()

    x = x.sort_values("ts").reset_index(drop=True)

    # Merge HTF 1D nativo -> 4H via merge_asof
    if cfg["use_htf"]:
        if htf_flags_1d_native is None or htf_flags_1d_native.empty:
            # Pela sua regra: se não for 1D baixado separado, não usar HTF.
            # Aqui a gente simplesmente não roda HTF (mas esse caso deve ser filtrado antes).
            return pd.DataFrame()

        tmp = pd.merge_asof(
            x[["ts"]],
            htf_flags_1d_native.sort_values("ts"),
            on="ts",
            direction="backward"
        )
        x["htf_trend_up"] = tmp["htf_trend_up"].fillna(False).astype(bool)
        x["htf_slope_up"] = tmp["htf_slope_up"].fillna(False).astype(bool)
    else:
        x["htf_trend_up"] = True
        x["htf_slope_up"] = True

    tick = float(x["close"].iloc[-1] * 0.0001)
    start_idx = max(SMA_LONG + 20, SLOPE_LOOKBACK + 5)

    trades = []

    for i in range(start_idx, len(x) - 10):
        # regime base no 4h
        if not (x["trend_up"].iloc[i] and x["slope_up"].iloc[i]):
            continue

        # HTF filter (1D nativo)
        if cfg["use_htf"]:
            if not x["htf_trend_up"].iloc[i]:
                continue
            if HTF_REQUIRE_SLOPE and (not x["htf_slope_up"].iloc[i]):
                continue

        pfr, dl, s82, s83 = check_signals(x, i)

        active = []
        if pfr:
            active.append("PFR")
        if dl and not pfr:
            active.append("DL")
        if s82:
            active.append("8.2")
        if s83:
            active.append("8.3")

        if not active:
            continue

        for setup in active:
            max_wait = 3 if setup in ["8.2", "8.3"] else 1

            entry = float(x["high"].iloc[i] + tick)
            stop = float(x["low"].iloc[i] - tick)

            filled = False
            fill_idx = -1

            # espera preenchimento
            for w in range(1, max_wait + 1):
                curr = x.iloc[i + w]

                if curr["high"] >= entry:
                    filled = True
                    fill_idx = i + w
                    break

                if setup in ["8.2", "8.3"]:
                    if not x["slope_up"].iloc[i + w]:
                        break
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

            # stretch features no candle do sinal (i)
            dist8 = x["dist_sma8"].iloc[i]
            dist80 = x["dist_sma80"].iloc[i]
            gap = x["sma_gap"].iloc[i]
            if np.isnan(dist8) or np.isnan(dist80) or np.isnan(gap):
                continue

            row = {
                "config": cfg["name"],
                "timeframe": "4h",
                "setup": setup,

                "signal_ts": x["ts"].iloc[i],
                "fill_ts": x["ts"].iloc[fill_idx],

                "entry": entry,
                "stop": stop,
                "risk": risk,

                "dist_sma8": float(dist8),
                "dist_sma80": float(dist80),
                "sma_gap": float(gap),

                "use_htf": cfg["use_htf"],
                "use_time_exit": cfg["use_time_exit"],
                "use_breakeven": cfg["use_breakeven"],
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

    return pd.DataFrame(trades)


# =============================
# CONFIGS (somente variáveis estruturais)
# =============================
def build_structural_configs(htf_available: bool):
    """
    Variáveis ON/OFF:
      - HTF (somente se 1D nativo estiver disponível)
      - time-exit
      - break-even
    """
    configs = []
    use_htf_options = [False, True] if htf_available else [False]

    for use_htf, use_time_exit, use_breakeven in product(use_htf_options, [False, True], [False, True]):
        name = f"htf{int(use_htf)}_tx{int(use_time_exit)}_be{int(use_breakeven)}"
        configs.append({
            "name": name,
            "use_htf": use_htf,
            "use_time_exit": use_time_exit,
            "use_breakeven": use_breakeven,
        })

    if len(configs) > MAX_CONFIGS:
        configs = configs[:MAX_CONFIGS]

    return configs


# =============================
# STRETCH FILTER (aplica depois)
# =============================
def apply_stretch_filter(trades_df, dist_sma8_max, dist_sma80_max, sma_gap_min):
    m = (
        (trades_df["dist_sma8"] <= dist_sma8_max) &
        (trades_df["dist_sma80"] <= dist_sma80_max) &
        (trades_df["sma_gap"] >= sma_gap_min)
    )
    return trades_df[m].copy()


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


def summary_by(trades_df, rr, by_cols):
    if trades_df.empty:
        return pd.DataFrame()

    k = rr_key(rr)
    rows = []
    for keys, g in trades_df.groupby(by_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(by_cols, keys)}
        row["Trades"] = len(g)
        row["WR"] = float(g[f"hit_{k}"].mean()) if len(g) else np.nan
        row["AvgR"] = float(g[f"R_{k}"].mean()) if len(g) else np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values(by_cols + ["Trades"], ascending=[True] * len(by_cols) + [False])


# =============================
# MAIN
# =============================
def main():
    os.makedirs("results", exist_ok=True)

    # 1) Baixa 4h (sempre)
    df_4h = fetch_4h(SYMBOL)
    if df_4h.empty:
        print("Erro: Sem dados 4h baixados.")
        pd.DataFrame({"status": ["no_4h_data"]}).to_csv(f"results/only4h_native1dhtf_{SYMBOL}.csv", index=False)
        return

    if SAVE_OHLCV:
        df_4h.to_csv(f"results/ohlcv_{SYMBOL}_4h.csv", index=False)

    # 2) Baixa 1d nativo (somente para HTF) - opcional
    htf_flags_1d_native = None
    htf_available = False

    if DOWNLOAD_NATIVE_1D_FOR_HTF:
        df_1d_native = fetch_1d_native(SYMBOL)
        if not df_1d_native.empty:
            if SAVE_OHLCV:
                df_1d_native.to_csv(f"results/ohlcv_{SYMBOL}_1d_native.csv", index=False)

            ind_1d = add_indicators(df_1d_native).sort_values("ts").reset_index(drop=True)
            htf_flags_1d_native = ind_1d[["ts", "trend_up", "slope_up"]].rename(columns={
                "trend_up": "htf_trend_up",
                "slope_up": "htf_slope_up",
            })
            htf_available = True
        else:
            print("Aviso: não foi possível baixar 1d nativo; configs com HTF serão DESABILITADAS.")
            htf_available = False
    else:
        print("DOWNLOAD_NATIVE_1D_FOR_HTF=0 => HTF desabilitado por regra.")
        htf_available = False

    # 3) Cria configs estruturais (com HTF somente se 1D nativo disponível)
    configs = build_structural_configs(htf_available=htf_available)
    print(f"ONLY 4H | HTF(native1d) available={htf_available} | configs={len(configs)}")

    # 4) Gera trades brutos por config (SEM stretch)
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
        pd.DataFrame({"status": ["no_trades"]}).to_csv(f"results/only4h_native1dhtf_{SYMBOL}.csv", index=False)
        return

    trades = pd.concat(all_trades, ignore_index=True)
    trades.to_csv(f"results/only4h_trades_raw_{SYMBOL}.csv", index=False)

    # 5) Split treino/teste global
    train, test, cutoff = split_train_test(trades)
    print(f"Cutoff treino/teste: {cutoff}")

    # 6) Stage 1: ranking configs (sem stretch)
    stage1_rows = []
    for cfg_name, _ in trades.groupby("config"):
        tr = train[train["config"] == cfg_name]
        te = test[test["config"] == cfg_name]
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

    print("\n=== Stage 1 (4h only, sem stretch) - ranking por Test AvgR @ RR="
          f"{RANK_RR} (min test trades={MIN_TRADES_FOR_RANK_TEST}) ===")
    print(tabulate(stage1, headers="keys", tablefmt="grid", showindex=False))

    eligible = stage1[stage1["Eligible"]].copy()
    if eligible.empty:
        top_cfgs = stage1.head(TOP_K_CONFIGS_FOR_SWEEP)["Config"].tolist()
    else:
        top_cfgs = eligible.head(TOP_K_CONFIGS_FOR_SWEEP)["Config"].tolist()

    print(f"\nConfigs selecionadas para Stage 2 (stretch sweep): {top_cfgs}")

    # 7) Stage 2: sweep stretch no treino e valida no teste
    stretch_grid = list(product(DIST_SMA8_MAX_LIST, DIST_SMA80_MAX_LIST, SMA_GAP_MIN_LIST))
    stage2_rows = []
    best_rows = []

    for cfg_name in top_cfgs:
        base_train = train[train["config"] == cfg_name].copy()
        base_test = test[test["config"] == cfg_name].copy()

        base_train_m = metrics_overall(base_train, RANK_RR)
        base_test_m = metrics_overall(base_test, RANK_RR)

        best = None
        for s8, s80, gap in stretch_grid:
            ftr = apply_stretch_filter(base_train, s8, s80, gap)
            m = metrics_overall(ftr, RANK_RR)

            stage2_rows.append({
                "Config": cfg_name,
                "dist_sma8_max": s8,
                "dist_sma80_max": s80,
                "sma_gap_min": gap,
                "Train Trades": m["Trades"],
                "Train WR": m["WR"],
                "Train AvgR": m["AvgR"],
            })

            if m["Trades"] < MIN_TRADES_FOR_SELECT_TRAIN:
                continue

            if best is None or (m["AvgR"] > best["Train AvgR"]):
                best = {
                    "Config": cfg_name,
                    "dist_sma8_max": s8,
                    "dist_sma80_max": s80,
                    "sma_gap_min": gap,
                    "Train Trades": m["Trades"],
                    "Train WR": m["WR"],
                    "Train AvgR": m["AvgR"],
                }

        # fallback: se nada bateu mínimo de trades no treino, pega o melhor mesmo assim
        if best is None:
            tmp = pd.DataFrame([r for r in stage2_rows if r["Config"] == cfg_name]).copy()
            tmp = tmp.sort_values(["Train AvgR", "Train Trades"], ascending=[False, False]).head(1)
            best = tmp.iloc[0].to_dict()

        best_test_filtered = apply_stretch_filter(
            base_test,
            best["dist_sma8_max"],
            best["dist_sma80_max"],
            best["sma_gap_min"]
        )
        best_test_m = metrics_overall(best_test_filtered, RANK_RR)

        best_rows.append({
            "Config": cfg_name,
            "BASE Train Trades": base_train_m["Trades"],
            "BASE Train AvgR": base_train_m["AvgR"],
            "BASE Test Trades": base_test_m["Trades"],
            "BASE Test AvgR": base_test_m["AvgR"],

            "BEST dist_sma8_max": best["dist_sma8_max"],
            "BEST dist_sma80_max": best["dist_sma80_max"],
            "BEST sma_gap_min": best["sma_gap_min"],

            "BEST Train Trades": best["Train Trades"],
            "BEST Train AvgR": best["Train AvgR"],

            "BEST Test Trades": best_test_m["Trades"],
            "BEST Test AvgR": best_test_m["AvgR"],
        })

        # diagnóstico no TESTE por setup (somente 4h)
        per_setup = summary_by(best_test_filtered, RANK_RR, ["setup"])
        per_setup.to_csv(f"results/only4h_stage2_best_per_setup_{SYMBOL}_{cfg_name}.csv", index=False)

    stage2 = pd.DataFrame(stage2_rows)
    stage2.to_csv(f"results/only4h_stage2_stretch_sweep_{SYMBOL}.csv", index=False)

    best_df = pd.DataFrame(best_rows).sort_values("BEST Test AvgR", ascending=False)
    best_df.to_csv(f"results/only4h_stage2_best_stretch_{SYMBOL}.csv", index=False)

    print("\n=== Stage 2 (4h only) - melhores stretch escolhidos no TREINO e validados no TESTE ===")
    print(tabulate(best_df, headers="keys", tablefmt="grid", showindex=False))

    with open(f"results/only4h_summary_{SYMBOL}.md", "w", encoding="utf-8") as f:
        f.write(f"# Only 4H Summary - {SYMBOL}\n\n")
        f.write("- Trades: somente 4h\n")
        f.write(f"- HTF: somente se 1D nativo baixar separado (available={htf_available})\n")
        f.write(f"- Train fraction: {TRAIN_FRACTION}\n")
        f.write(f"- Rank RR: {RANK_RR}\n")
        f.write(f"- Min test trades rank: {MIN_TRADES_FOR_RANK_TEST}\n")
        f.write(f"- Min train trades select: {MIN_TRADES_FOR_SELECT_TRAIN}\n")
        f.write(f"- Time exit bars (4h): {TIME_EXIT_BARS}\n")
        f.write(f"- Break-even fraction of target: {BE_TARGET_FRACTION}\n")
        f.write(f"- HTF require slope: {HTF_REQUIRE_SLOPE}\n\n")
        f.write("## Stage 1 (Structural, sem stretch)\n\n")
        f.write(tabulate(stage1, headers="keys", tablefmt="pipe", showindex=False))
        f.write("\n\n## Stage 2 (Best stretch per selected config)\n\n")
        f.write(tabulate(best_df, headers="keys", tablefmt="pipe", showindex=False))


if __name__ == "__main__":
    main()
