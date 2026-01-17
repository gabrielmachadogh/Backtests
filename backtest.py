import os
import time
import requests
import numpy as np
import pandas as pd
from tabulate import tabulate
from itertools import product

# =========================================================
# PFR ONLY - 4H - STEP 4 TEST SCRIPT
#
# Objetivo (seu "passo 4"):
# 1) Aumentar histórico (mais candles 4h)
# 2) Testar impacto de:
#    - TIME EXIT ON vs OFF
#    - BREAK EVEN ON vs OFF (sem precisar "misturar tudo" manualmente)
# 3) Rodar em múltiplos símbolos (BTC + outros)
#
# O script roda, para cada símbolo:
# - 8 configs de filtros A/B/C (mesmos que você usou)
# - 4 perfis de saída (exit profiles):
#   E00: time_exit=0, breakeven=0
#   E10: time_exit=1, breakeven=0
#   E01: time_exit=0, breakeven=1
#   E11: time_exit=1, breakeven=1
#
# E gera:
# - tabela Stage1 por símbolo + exit_profile (para RR 1, 1.5, 2, 3)
# - CSVs em results/
# - um MASTER CSV juntando tudo
#
# Definições:
# - "apontando para cima" = SMA(t) > SMA(t-1)
#
# Filtros:
# A: sma8>sma80 AND sma8_up AND sma80_up
# B: sma9_up
# C: sma20>sma200 AND sma20_up
#
# Setup:
# PFR:
#   low[i] < low[i-1] AND low[i] < low[i-2] AND close[i] > close[i-1]
#
# Execução:
# - entry = high[i] + tick
# - stop  = low[i] - tick
# - fill: no máximo 1 candle à frente (PFR)
#
# =========================================================

# =============================
# CONFIG
# =============================
BASE_URL = os.getenv("MEXC_CONTRACT_BASE_URL", "https://contract.mexc.com/api/v1")
DEBUG = os.getenv("DEBUG", "0") == "1"

# Símbolos
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC_USDT,ETH_USDT,SOL_USDT").split(",") if s.strip()]

# RR
RRS = [float(x) for x in os.getenv("RRS", "1.0,1.5,2.0,3.0").split(",")]
RANK_RR = float(os.getenv("RANK_RR", "1.5"))

# Download 4h (AUMENTADO por padrão)
MAX_BARS_FETCH_4H = int(os.getenv("MAX_BARS_FETCH_4H", "40000"))
WINDOW_DAYS_4H = int(os.getenv("WINDOW_DAYS_4H", "365"))  # mais histórico por chunk

SAVE_OHLCV = os.getenv("SAVE_OHLCV", "1") == "1"

# PFR fill
MAX_WAIT_FILL = int(os.getenv("MAX_WAIT_FILL", "1"))  # PFR: 1

# Horizon
MAX_HOLD_BARS = int(os.getenv("MAX_HOLD_BARS", "50"))

# Break-even
BE_TARGET_FRACTION = float(os.getenv("BE_TARGET_FRACTION", "0.70"))

# Split treino/teste
TRAIN_FRACTION = float(os.getenv("TRAIN_FRACTION", "0.70"))
MIN_TRADES_FOR_RANK_TEST = int(os.getenv("MIN_TRADES_FOR_RANK_TEST", "80"))

# Controle de execução (se ficar pesado)
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "50"))


# =============================
# HELPERS
# =============================
def rr_key(rr: float) -> str:
    if float(rr).is_integer():
        return str(int(rr))
    return str(rr)


def http_get_json(url, params=None, tries=3, sleep_s=1.0):
    headers = {"User-Agent": "backtest-bot/1.0"}
    last_err = None
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
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
    """
    Baixa candles 4h via janelas (start/end). Tenta Hour4 e Min240.
    """
    interval_candidates = ["Hour4", "Min240"]
    print(f"Baixando histórico 4h para {symbol}...")

    end_ts = int(time.time())
    step = int(WINDOW_DAYS_4H) * 86400
    all_dfs = []
    total = 0

    for n in range(400):
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
            print(f"[{symbol} 4h] used_interval={used_interval}")

        if df.empty:
            break

        all_dfs.append(df)
        total += len(df)

        first_ts_val = int(df.iloc[0]["ts"].timestamp())
        if first_ts_val >= end_ts:
            break
        end_ts = first_ts_val - 1

        time.sleep(0.15)

    if not all_dfs:
        return pd.DataFrame()

    full_df = (
        pd.concat(all_dfs)
        .drop_duplicates("ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )

    print(f"{symbol}: {len(full_df)} candles 4h baixados.")
    return full_df


# =============================
# INDICATORS / FILTERS
# =============================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["sma8"] = x["close"].rolling(8).mean()
    x["sma9"] = x["close"].rolling(9).mean()
    x["sma20"] = x["close"].rolling(20).mean()
    x["sma80"] = x["close"].rolling(80).mean()
    x["sma200"] = x["close"].rolling(200).mean()

    x["sma8_up"] = x["sma8"] > x["sma8"].shift(1)
    x["sma9_up"] = x["sma9"] > x["sma9"].shift(1)
    x["sma20_up"] = x["sma20"] > x["sma20"].shift(1)
    x["sma80_up"] = x["sma80"] > x["sma80"].shift(1)

    x["f_8_80"] = (x["sma8"] > x["sma80"]) & x["sma8_up"] & x["sma80_up"]
    x["f_9"] = x["sma9_up"]
    x["f_20_200"] = (x["sma20"] > x["sma200"]) & x["sma20_up"]

    return x


# =============================
# SETUP PFR
# =============================
def is_pfr(x, i) -> bool:
    return (
        (x["low"].iloc[i] < x["low"].iloc[i - 1])
        and (x["low"].iloc[i] < x["low"].iloc[i - 2])
        and (x["close"].iloc[i] > x["close"].iloc[i - 1])
    )


# =============================
# TRADE SIM
# =============================
def simulate_trade_per_rr(x, fill_idx, entry, initial_stop, rr, exit_profile):
    """
    exit_profile:
      - use_time_exit: bool
      - use_breakeven: bool
    """
    risk = entry - initial_stop
    if risk <= 0 or np.isnan(risk):
        return False, "invalid", np.nan, pd.NaT

    stop = initial_stop
    target = entry + risk * rr

    be_trigger = entry + (target - entry) * BE_TARGET_FRACTION
    be_pending = False
    be_active = False

    last_idx = min(fill_idx + max(MAX_HOLD_BARS, 1) - 1, len(x) - 1)

    for k in range(fill_idx, last_idx + 1):
        c = x.iloc[k]

        # BE efetivo no início do candle seguinte ao trigger
        if exit_profile["use_breakeven"] and be_pending and not be_active:
            stop = entry
            be_active = True
            be_pending = False

        # ordem conservadora intrabar: stop -> target -> trigger
        if c["low"] <= stop:
            r = (stop - entry) / risk
            return False, "stop" if stop < entry else "breakeven_stop", float(r), c["ts"]

        if c["high"] >= target:
            return True, "target", float(rr), c["ts"]

        if exit_profile["use_breakeven"] and (not be_active) and (c["high"] >= be_trigger):
            be_pending = True

    if exit_profile["use_time_exit"]:
        exit_px = float(x["close"].iloc[last_idx])
        r = (exit_px - entry) / risk
        return False, "time_exit", float(r), x["ts"].iloc[last_idx]

    return False, "timeout_loss", -1.0, x["ts"].iloc[last_idx]


# =============================
# BACKTEST PFR
# =============================
def run_pfr_backtest(x: pd.DataFrame, filter_cfg: dict, exit_profile: dict) -> pd.DataFrame:
    """
    filter_cfg:
      - name (a0_b1_c0 etc)
      - use_8_80, use_9, use_20_200
    exit_profile:
      - name (E00/E10/E01/E11)
      - use_time_exit, use_breakeven
    """
    tick = float(x["close"].iloc[-1] * 0.0001)

    start_idx = max(200 + 2, 80 + 2, 20 + 2, 9 + 2, 8 + 2, 5)
    trades = []

    for i in range(start_idx, len(x) - 2):
        # filtros
        if filter_cfg["use_8_80"] and (not bool(x["f_8_80"].iloc[i])):
            continue
        if filter_cfg["use_9"] and (not bool(x["f_9"].iloc[i])):
            continue
        if filter_cfg["use_20_200"] and (not bool(x["f_20_200"].iloc[i])):
            continue

        if not is_pfr(x, i):
            continue

        entry = float(x["high"].iloc[i] + tick)
        stop = float(x["low"].iloc[i] - tick)

        # fill (1 candle)
        filled = False
        fill_idx = -1
        for w in range(1, MAX_WAIT_FILL + 1):
            if i + w >= len(x):
                break
            if x["high"].iloc[i + w] >= entry:
                filled = True
                fill_idx = i + w
                break

        if not filled:
            continue

        risk = entry - stop
        if risk <= 0 or np.isnan(risk):
            continue

        row = {
            "filter_cfg": filter_cfg["name"],
            "exit_profile": exit_profile["name"],
            "setup": "PFR",
            "signal_ts": x["ts"].iloc[i],
            "fill_ts": x["ts"].iloc[fill_idx],
            "entry": entry,
            "stop": stop,
            "risk": risk,
            "use_8_80": filter_cfg["use_8_80"],
            "use_9": filter_cfg["use_9"],
            "use_20_200": filter_cfg["use_20_200"],
            "use_time_exit": exit_profile["use_time_exit"],
            "use_breakeven": exit_profile["use_breakeven"],
        }

        for rr in RRS:
            hit, exit_type, r_res, exit_ts = simulate_trade_per_rr(
                x=x, fill_idx=fill_idx, entry=entry, initial_stop=stop, rr=rr, exit_profile=exit_profile
            )
            k = rr_key(rr)
            row[f"hit_{k}"] = bool(hit)
            row[f"exit_type_{k}"] = exit_type
            row[f"R_{k}"] = float(r_res) if r_res is not None else np.nan
            row[f"exit_ts_{k}"] = exit_ts

        trades.append(row)

    return pd.DataFrame(trades)


# =============================
# SPLIT / METRICS
# =============================
def split_train_test(trades_df: pd.DataFrame):
    if trades_df.empty:
        return trades_df.copy(), trades_df.copy(), pd.NaT
    trades_df = trades_df.sort_values("fill_ts").reset_index(drop=True)
    cutoff = trades_df["fill_ts"].quantile(TRAIN_FRACTION)
    train = trades_df[trades_df["fill_ts"] <= cutoff].copy()
    test = trades_df[trades_df["fill_ts"] > cutoff].copy()
    return train, test, cutoff


def metrics_for_rr(trades_df: pd.DataFrame, rr: float):
    if trades_df.empty:
        return {"Trades": 0, "WR": np.nan, "AvgR": np.nan}
    k = rr_key(rr)
    n = len(trades_df)
    wr = float(trades_df[f"hit_{k}"].mean()) if n else np.nan
    avgr = float(trades_df[f"R_{k}"].mean()) if n else np.nan
    return {"Trades": n, "WR": wr, "AvgR": avgr}


# =============================
# MAIN
# =============================
def main():
    os.makedirs("results", exist_ok=True)

    # configs de filtro A/B/C (8)
    filter_cfgs = []
    for use_8_80, use_9, use_20_200 in product([False, True], repeat=3):
        name = f"a{int(use_8_80)}_b{int(use_9)}_c{int(use_20_200)}"
        filter_cfgs.append({
            "name": name,
            "use_8_80": use_8_80,
            "use_9": use_9,
            "use_20_200": use_20_200,
        })

    # perfis de saída (4)
    exit_profiles = [
        {"name": "E00", "use_time_exit": False, "use_breakeven": False},
        {"name": "E10", "use_time_exit": True,  "use_breakeven": False},
        {"name": "E01", "use_time_exit": False, "use_breakeven": True},
        {"name": "E11", "use_time_exit": True,  "use_breakeven": True},
    ]

    master_stage1 = []
    master_best = []

    for sym in SYMBOLS[:MAX_SYMBOLS]:
        df = fetch_history_4h(sym)
        if df.empty:
            print(f"{sym}: sem dados, pulando.")
            continue

        if SAVE_OHLCV:
            df.to_csv(f"results/ohlcv_{sym}_4h.csv", index=False)

        x = add_indicators(df).sort_values("ts").reset_index(drop=True)

        all_trades = []
        for ep in exit_profiles:
            for fc in filter_cfgs:
                t = run_pfr_backtest(x, fc, ep)
                if not t.empty:
                    t.insert(0, "symbol", sym)
                    all_trades.append(t)

        if not all_trades:
            print(f"{sym}: 0 trades PFR em todos configs.")
            continue

        trades = pd.concat(all_trades, ignore_index=True)
        trades.to_csv(f"results/pfr4h_trades_{sym}.csv", index=False)

        # Stage1 por exit_profile + filter_cfg
        rows = []
        for (ep_name, fc_name), g in trades.groupby(["exit_profile", "filter_cfg"]):
            train, test, cutoff = split_train_test(g)

            row = {
                "Symbol": sym,
                "ExitProfile": ep_name,
                "FilterCfg": fc_name,
                "Train Trades": len(train),
                "Test Trades": len(test),
                "Eligible": len(test) >= MIN_TRADES_FOR_RANK_TEST,
            }

            for rr in RRS:
                k = rr_key(rr)
                m_tr = metrics_for_rr(train, rr)
                m_te = metrics_for_rr(test, rr)
                row[f"Train WR {k}"] = m_tr["WR"]
                row[f"Train AvgR {k}"] = m_tr["AvgR"]
                row[f"Test WR {k}"] = m_te["WR"]
                row[f"Test AvgR {k}"] = m_te["AvgR"]

            rows.append(row)

        stage1 = pd.DataFrame(rows)
        rk = rr_key(RANK_RR)
        stage1 = stage1.sort_values(
            ["Eligible", f"Test AvgR {rk}", "Test Trades"],
            ascending=[False, False, False]
        )

        stage1.to_csv(f"results/pfr4h_stage1_{sym}.csv", index=False)
        master_stage1.append(stage1)

        # imprime um resumo curto (top 8)
        print("\n" + "=" * 100)
        print(f"PFR 4H - {sym}")
        print(f"RRs={RRS} | RankRR={RANK_RR} | TrainFraction={TRAIN_FRACTION}")
        print("ExitProfiles: E00(no tx/no be), E10(tx), E01(be), E11(tx+be)")
        print("Filters: A(8/80 up & 8>80), B(sma9_up), C(sma20_up & 20>200)")
        print(tabulate(stage1.head(12), headers="keys", tablefmt="grid", showindex=False))

        # salva best por perfil (para bater o olho)
        best_per_profile = []
        for ep_name, gg in stage1.groupby("ExitProfile"):
            best_per_profile.append(gg.head(1))
        if best_per_profile:
            best_df = pd.concat(best_per_profile, ignore_index=True)
            best_df.insert(0, "RankRR", RANK_RR)
            best_df.to_csv(f"results/pfr4h_best_by_exitprofile_{sym}.csv", index=False)
            master_best.append(best_df)

    if master_stage1:
        master = pd.concat(master_stage1, ignore_index=True)
        master.to_csv("results/pfr4h_stage1_MASTER.csv", index=False)

    if master_best:
        best_master = pd.concat(master_best, ignore_index=True)
        best_master.to_csv("results/pfr4h_best_by_exitprofile_MASTER.csv", index=False)

    with open("results/pfr4h_step4_summary.md", "w", encoding="utf-8") as f:
        f.write("# PFR 4H - Step 4 Summary\n\n")
        f.write(f"- SYMBOLS={SYMBOLS}\n")
        f.write(f"- WINDOW_DAYS_4H={WINDOW_DAYS_4H}\n")
        f.write(f"- MAX_BARS_FETCH_4H={MAX_BARS_FETCH_4H}\n")
        f.write(f"- RRs={RRS}\n")
        f.write(f"- Rank RR={RANK_RR}\n")
        f.write(f"- Train fraction={TRAIN_FRACTION}\n")
        f.write(f"- MIN_TRADES_FOR_RANK_TEST={MIN_TRADES_FOR_RANK_TEST}\n")
        f.write(f"- MAX_HOLD_BARS={MAX_HOLD_BARS}\n")
        f.write(f"- BE_TARGET_FRACTION={BE_TARGET_FRACTION}\n\n")
        f.write("Outputs:\n")
        f.write("- results/pfr4h_trades_<SYMBOL>.csv\n")
        f.write("- results/pfr4h_stage1_<SYMBOL>.csv\n")
        f.write("- results/pfr4h_best_by_exitprofile_<SYMBOL>.csv\n")
        f.write("- results/pfr4h_stage1_MASTER.csv\n")
        f.write("- results/pfr4h_best_by_exitprofile_MASTER.csv\n")

    print("\nArquivos gerados em results/:")
    print("- pfr4h_stage1_MASTER.csv")
    print("- pfr4h_best_by_exitprofile_MASTER.csv")
    print("- pfr4h_trades_<SYMBOL>.csv / pfr4h_stage1_<SYMBOL>.csv")


if __name__ == "__main__":
    main()
