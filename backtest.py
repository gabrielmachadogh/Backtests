import os
import time
import requests
import numpy as np
import pandas as pd
from tabulate import tabulate
from itertools import product

# =========================================================
# ONLY 4H TRADING (NO HTF) + MULTI-SCENARIO RUN
#
# Objetivo:
# - Baixar APENAS candles de 4h
# - Rodar, na MESMA execução, estes cenários e imprimir tabelas Stage1:
#   1) Só MA_LADDER (rank em MA_LADDER)
#   2) Só clássicos (PFR/DL/8.2/8.3) rank ALL
#   3) Clássicos por setup: rank PFR, DL, 8.2, 8.3
#
# Agora a tabela Stage1 mostra WR e AvgR para RR = 1, 1.5, 2 e 3
# (tanto no TREINO quanto no TESTE), e o ranking continua pelo RANK_RR.
# =========================================================

# =============================
# CONFIG (env-friendly)
# =============================
BASE_URL = os.getenv("MEXC_CONTRACT_BASE_URL", "https://contract.mexc.com/api/v1")
SYMBOL = os.getenv("SYMBOL", "BTC_USDT")
DEBUG = os.getenv("DEBUG", "0") == "1"

# Ladder MAs
LADDER_MAS = [10, 15, 20, 25, 30, 35, 40, 45]
LADDER_DISARM_ON_CLOSE_BELOW = 40  # desarma se close < SMA40

# RR (default já inclui 3.0)
RRS = [float(x) for x in os.getenv("RRS", "1.0,1.5,2.0,3.0").split(",")]
RANK_RR = float(os.getenv("RANK_RR", "1.5"))

# Download 4h
MAX_BARS_FETCH_4H = int(os.getenv("MAX_BARS_FETCH_4H", "12000"))
WINDOW_DAYS_4H = int(os.getenv("WINDOW_DAYS_4H", "90"))

SAVE_OHLCV = os.getenv("SAVE_OHLCV", "1") == "1"

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


def filter_for_rank(df: pd.DataFrame, rank_setup: str) -> pd.DataFrame:
    if df.empty:
        return df
    if (rank_setup or "").upper() == "ALL":
        return df
    return df[df["setup"] == rank_setup].copy()


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


def fetch_history_4h(symbol):
    print(f"Baixando histórico 4h para {symbol}...")
    interval_candidates = ["Hour4", "Min240"]

    end_ts = int(time.time())
    step = int(WINDOW_DAYS_4H) * 86400
    all_dfs = []
    total = 0

    for n in range(200):
        if total >= MAX_BARS_FETCH_4H:
            break

        start_ts = end_ts - step

        df = pd.DataFrame()
        used_interval = None

        for interval in interval_candidates:
            url = f"{BASE_URL}/contract/kline/{symbol}"
            params = {"interval": interval, "start": start_ts, "end": end_ts}
            payload = http_get_json(url, params=params)
            df = parse_kline(payload)
            if not df.empty:
                used_interval = interval
                break

        if DEBUG and n == 0:
            print(f"[4h] candidates={interval_candidates} used={used_interval}")

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
    print(f"Total baixado: {len(full_df)} candles (4h).")
    return full_df


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
    Clássicos só rodam se:
      - alinhamento strict
      - sma10_up
      - close>sma10 opcional (cfg['req_close_gt10'])
    """
    if not ladder_aligned_strict(x, i):
        return False
    if not bool(x["sma10_up"].iloc[i]):
        return False
    if cfg.get("req_close_gt10", False) and (not bool(x["close_gt_sma10"].iloc[i])):
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

        if cfg.get("use_breakeven", False) and be_pending and not be_active:
            stop = entry
            be_active = True
            be_pending = False

        if c["low"] <= stop:
            r = (stop - entry) / risk
            return False, "stop" if stop < entry else "breakeven_stop", float(r), c["ts"]

        if c["high"] >= target:
            return True, "target", float(rr), c["ts"]

        if cfg.get("use_breakeven", False) and (not be_active) and (c["high"] >= be_trigger):
            be_pending = True

    if cfg.get("use_time_exit", False):
        exit_px = float(x["close"].iloc[last_idx])
        r = (exit_px - entry) / risk
        return False, "time_exit", float(r), x["ts"].iloc[last_idx]

    return False, "timeout_loss", -1.0, x["ts"].iloc[last_idx]


# =============================
# BACKTEST CORE (gera trades brutos)
# =============================
def run_backtest_generate_trades_4h(x, cfg, setups):
    tick = float(x["close"].iloc[-1] * 0.0001)
    start_idx = max(max(LADDER_MAS) + 5, 5)

    trades = []

    ladder_enabled = "MA_LADDER" in setups
    classics_enabled = any(s in setups for s in ["PFR", "DL", "8.2", "8.3"])

    # MA_LADDER state
    ladder_armed = False
    ladder_armed_ts = pd.NaT

    ladder_touched_period = None
    ladder_touch_idx = None
    ladder_stop_fixed = None  # STOP travado

    ladder_pending = False
    ladder_pending_entry = None
    ladder_pending_stop = None
    ladder_pending_signal_idx = None
    ladder_pending_touch_period = None
    ladder_pending_touch_idx = None

    for i in range(start_idx, len(x)):
        # 1) fill ladder
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

                    "use_time_exit": cfg.get("use_time_exit", False),
                    "use_breakeven": cfg.get("use_breakeven", False),
                    "req_close_gt10": cfg.get("req_close_gt10", np.nan),
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

                # reset ladder
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

        # 2) classics
        if classics_enabled and i >= 2 and classic_allowed_regime(x, i, cfg):
            pfr, dl, s82, s83 = check_signals_classic(x, i)

            active = []
            if pfr and "PFR" in setups:
                active.append("PFR")
            if dl and (not pfr) and "DL" in setups:
                active.append("DL")
            if s82 and "8.2" in setups:
                active.append("8.2")
            if s83 and "8.3" in setups:
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

                    "use_time_exit": cfg.get("use_time_exit", False),
                    "use_breakeven": cfg.get("use_breakeven", False),
                    "req_close_gt10": cfg.get("req_close_gt10", np.nan),
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

        # 3) ladder state machine
        if ladder_enabled:
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

            if (not ladder_armed) and ladder_armed_condition(x, i):
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
                touched = deepest_touched_ladder_ma(x, i)

                if touched is not None and (ladder_touched_period is None or touched > ladder_touched_period):
                    ladder_touched_period = touched
                    ladder_touch_idx = i

                    below_p = ladder_below_period(ladder_touched_period)
                    below_ma_val = x[f"sma{below_p}"].iloc[i]
                    ladder_stop_fixed = None if np.isnan(below_ma_val) else float(below_ma_val - tick)

                if ladder_touched_period is not None and ladder_stop_fixed is not None:
                    ladder_pending = True
                    ladder_pending_entry = float(x["high"].iloc[i] + tick)  # rola entry
                    ladder_pending_stop = float(ladder_stop_fixed)          # stop travado
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
def build_structural_configs(include_c10: bool):
    configs = []
    if include_c10:
        for use_time_exit, use_breakeven, req_close_gt10 in product([False, True], [False, True], [False, True]):
            name = f"tx{int(use_time_exit)}_be{int(use_breakeven)}_c10{int(req_close_gt10)}"
            configs.append({
                "name": name,
                "use_time_exit": use_time_exit,
                "use_breakeven": use_breakeven,
                "req_close_gt10": req_close_gt10,
            })
    else:
        for use_time_exit, use_breakeven in product([False, True], [False, True]):
            name = f"tx{int(use_time_exit)}_be{int(use_breakeven)}"
            configs.append({
                "name": name,
                "use_time_exit": use_time_exit,
                "use_breakeven": use_breakeven,
            })

    if len(configs) > MAX_CONFIGS:
        configs = configs[:MAX_CONFIGS]
    return configs


# =============================
# METRICS / SPLIT
# =============================
def split_train_test(trades_df):
    if trades_df.empty:
        return trades_df.copy(), trades_df.copy(), pd.NaT
    trades_df = trades_df.sort_values("fill_ts").reset_index(drop=True)
    cutoff = trades_df["fill_ts"].quantile(TRAIN_FRACTION)
    train = trades_df[trades_df["fill_ts"] <= cutoff].copy()
    test = trades_df[trades_df["fill_ts"] > cutoff].copy()
    return train, test, cutoff


def metrics_for_rr(trades_df, rr):
    if trades_df.empty:
        return {"Trades": 0, "WR": np.nan, "AvgR": np.nan}
    k = rr_key(rr)
    trades = len(trades_df)
    wr = float(trades_df[f"hit_{k}"].mean()) if trades > 0 else np.nan
    avgr = float(trades_df[f"R_{k}"].mean()) if trades > 0 else np.nan
    return {"Trades": trades, "WR": wr, "AvgR": avgr}


# =============================
# SCENARIO RUNNER
# =============================
def run_scenario(x, scenario_name: str, setups: list, rank_setup: str):
    os.makedirs("results", exist_ok=True)

    classics_in_scenario = any(s in setups for s in ["PFR", "DL", "8.2", "8.3"])
    configs = build_structural_configs(include_c10=classics_in_scenario)

    all_trades = []
    for cfg in configs:
        t = run_backtest_generate_trades_4h(x=x, cfg=cfg, setups=setups)
        if not t.empty:
            t.insert(0, "scenario", scenario_name)
            all_trades.append(t)

    if not all_trades:
        return pd.DataFrame(), pd.DataFrame()

    trades = pd.concat(all_trades, ignore_index=True)

    train, test, cutoff = split_train_test(trades)
    train_rank = filter_for_rank(train, rank_setup)
    test_rank = filter_for_rank(test, rank_setup)

    rows = []
    for cfg_name, _ in trades.groupby("config"):
        tr = train_rank[train_rank["config"] == cfg_name]
        te = test_rank[test_rank["config"] == cfg_name]

        row = {
            "Scenario": scenario_name,
            "RankSetup": rank_setup,
            "Config": cfg_name,
            "Train Trades": len(tr),
            "Test Trades": len(te),
            "Eligible": (len(te) >= MIN_TRADES_FOR_RANK_TEST),
        }

        # métricas para cada RR solicitado
        for rr in RRS:
            k = rr_key(rr)
            m_tr = metrics_for_rr(tr, rr)
            m_te = metrics_for_rr(te, rr)
            row[f"Train WR {k}"] = m_tr["WR"]
            row[f"Train AvgR {k}"] = m_tr["AvgR"]
            row[f"Test WR {k}"] = m_te["WR"]
            row[f"Test AvgR {k}"] = m_te["AvgR"]

        rows.append(row)

    stage1 = pd.DataFrame(rows)

    # Ranking principal pelo Test AvgR do RANK_RR
    rk = rr_key(RANK_RR)
    stage1 = stage1.sort_values(
        ["Eligible", f"Test AvgR {rk}", "Test Trades"],
        ascending=[False, False, False]
    )

    trades.to_csv(f"results/scenario_trades_{SYMBOL}_{scenario_name}.csv", index=False)
    stage1.to_csv(f"results/scenario_stage1_{SYMBOL}_{scenario_name}.csv", index=False)

    # Print para você colar aqui
    print("\n" + "=" * 90)
    print(f"SCENARIO: {scenario_name} | SETUPS={setups} | RANK_SETUP={rank_setup} | RANK_RR={RANK_RR}")
    print(f"Cutoff train/test: {cutoff}")
    cols = ["Config", "Train Trades", "Test Trades"] + \
           [f"Test AvgR {rr_key(RANK_RR)}"] + \
           [f"Test WR {rr_key(rr)}" for rr in RRS] + \
           [f"Test AvgR {rr_key(rr)}" for rr in RRS]
    # Mostra tabela completa (mas numa ordem “boa de ler”)
    show = stage1.copy()
    # se algum RR faltar por env, garante
    show = show[[c for c in show.columns if c in stage1.columns]]
    print(tabulate(stage1.drop(columns=["Scenario", "RankSetup"]), headers="keys", tablefmt="grid", showindex=False))

    return trades, stage1


# =============================
# MAIN
# =============================
def main():
    os.makedirs("results", exist_ok=True)

    df_4h = fetch_history_4h(SYMBOL)
    if df_4h.empty:
        print("Erro: Sem dados 4h baixados.")
        pd.DataFrame({"status": ["no_4h_data"]}).to_csv(f"results/no_data_{SYMBOL}.csv", index=False)
        return

    if SAVE_OHLCV:
        df_4h.to_csv(f"results/ohlcv_{SYMBOL}_4h.csv", index=False)

    x = add_indicators(df_4h).sort_values("ts").reset_index(drop=True)

    scenarios = [
        ("LADDER_ONLY", ["MA_LADDER"], "MA_LADDER"),
        ("CLASSICS_ALL", ["PFR", "DL", "8.2", "8.3"], "ALL"),
        ("CLASSICS_PFR", ["PFR", "DL", "8.2", "8.3"], "PFR"),
        ("CLASSICS_DL", ["PFR", "DL", "8.2", "8.3"], "DL"),
        ("CLASSICS_8_2", ["PFR", "DL", "8.2", "8.3"], "8.2"),
        ("CLASSICS_8_3", ["PFR", "DL", "8.2", "8.3"], "8.3"),
    ]

    master_stage1 = []
    for name, setups, rank_setup in scenarios:
        _, s1 = run_scenario(x=x, scenario_name=name, setups=setups, rank_setup=rank_setup)
        if not s1.empty:
            master_stage1.append(s1)

    if master_stage1:
        master = pd.concat(master_stage1, ignore_index=True)
        master.to_csv(f"results/scenario_stage1_{SYMBOL}_MASTER.csv", index=False)

        with open(f"results/scenario_summary_{SYMBOL}.md", "w", encoding="utf-8") as f:
            f.write(f"# Scenario Summary - {SYMBOL}\n\n")
            f.write(f"- RRs: {RRS}\n")
            f.write(f"- Rank RR: {RANK_RR}\n")
            f.write(f"- Train fraction: {TRAIN_FRACTION}\n")
            f.write(f"- Time exit bars (4h): {TIME_EXIT_BARS}\n")
            f.write(f"- Break-even fraction of target: {BE_TARGET_FRACTION}\n\n")
            f.write("Generated CSVs:\n")
            f.write(f"- results/scenario_stage1_{SYMBOL}_*.csv\n")
            f.write(f"- results/scenario_trades_{SYMBOL}_*.csv\n")
    else:
        print("Nenhum cenário gerou trades.")


if __name__ == "__main__":
    main()
