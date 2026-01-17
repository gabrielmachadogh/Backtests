import os
import time
import requests
import numpy as np
import pandas as pd
from tabulate import tabulate

# =============================
# CONFIG
# =============================
BASE_URL = os.getenv("MEXC_CONTRACT_BASE_URL", "https://contract.mexc.com/api/v1")
SYMBOL = os.getenv("SYMBOL", "BTC_USDT")
DEBUG = os.getenv("DEBUG", "0") == "1"

TIMEFRAMES = ["1h", "2h", "4h", "1d"]
SETUPS = ["PFR", "DL", "8.2", "8.3"]

SMA_SHORT = 8
SMA_LONG = 80
SLOPE_LOOKBACK = 8

RRS = [1.0, 1.5, 2.0]
MAX_BARS_FETCH = 18000
WINDOW_DAYS = 30

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
    """
    Suporta:
      - payload["data"] como dict de listas (comum na MEXC)
      - payload["data"] como lista de listas
      - payload["data"] como lista de dicts
    Retorna DF com colunas: ts, open, high, low, close, volume
    """
    if not payload or not isinstance(payload, dict):
        return pd.DataFrame()

    data = payload.get("data", None)
    if data is None:
        return pd.DataFrame()

    try:
        # Caso 1: dict de listas
        if isinstance(data, dict):
            df = pd.DataFrame(data)

        # Caso 2: lista de listas
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            # esperando: [time, open, high, low, close, vol] (ou similar)
            df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "vol"])

        # Caso 3: lista de dicts
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)

        else:
            return pd.DataFrame()

    except Exception:
        return pd.DataFrame()

    # Normalização de nomes
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

    # Numéricos
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Timestamp
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    if df.empty:
        return df

    # ms vs s
    if df["ts"].iloc[0] > 1e11:
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    else:
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    return df.sort_values("ts").reset_index(drop=True)


def fetch_history(symbol):
    print(f"Baixando histórico 1h para {symbol}...")
    all_dfs = []
    end_ts = int(time.time())
    step = WINDOW_DAYS * 86400

    total = 0
    for n in range(50):  # limite de segurança
        if total >= MAX_BARS_FETCH:
            break

        start_ts = end_ts - step
        url = f"{BASE_URL}/contract/kline/{symbol}"
        params = {"interval": "Min60", "start": start_ts, "end": end_ts}
        payload = http_get_json(url, params=params)

        if DEBUG and n == 0:
            print("Exemplo payload keys:", list(payload.keys()) if isinstance(payload, dict) else type(payload))

        df = parse_kline(payload)
        if df.empty:
            break

        all_dfs.append(df)
        total += len(df)

        # recua o tempo para antes do primeiro candle retornado
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
    print(f"Total baixado: {len(full_df)} candles.")
    return full_df


def resample(df, rule):
    if df.empty:
        return df
    if rule == "1h":
        return df.copy()

    mapping = {"2h": "2h", "4h": "4h", "1d": "1d"}
    if rule not in mapping:
        return df

    df = df.set_index("ts")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    res = df.resample(mapping[rule]).agg(agg).dropna()
    return res.reset_index()

# =============================
# LOGIC
# =============================
def add_indicators(df):
    x = df.copy()
    x["sma_s"] = x["close"].rolling(SMA_SHORT).mean()
    x["sma_l"] = x["close"].rolling(SMA_LONG).mean()

    # Tendência
    x["trend_up"] = (x["close"] > x["sma_s"]) & (x["close"] > x["sma_l"]) & (x["sma_s"] > x["sma_l"])

    # Slope (sma_s atual maior que o máximo dos últimos N valores (sem contar o atual))
    x["slope_up"] = x["sma_s"] > x["sma_s"].shift(1).rolling(SLOPE_LOOKBACK).max()

    return x


def check_signals(x, i):
    # PFR
    pfr = (
        (x["low"].iloc[i] < x["low"].iloc[i - 1])
        and (x["low"].iloc[i] < x["low"].iloc[i - 2])
        and (x["close"].iloc[i] > x["close"].iloc[i - 1])
    )

    # DL
    dl = (x["low"].iloc[i] < x["low"].iloc[i - 1]) and (x["low"].iloc[i] < x["low"].iloc[i - 2])

    # 8.2
    s82 = x["close"].iloc[i] < x["low"].iloc[i - 1]

    # 8.3
    s83 = (
        (x["close"].iloc[i] < x["close"].iloc[i - 2])
        and (x["close"].iloc[i - 1] < x["close"].iloc[i - 2])
        and (x["close"].iloc[i - 1] >= x["low"].iloc[i - 2])
    )

    return pfr, dl, s82, s83


def run_backtest(df, tf):
    x = add_indicators(df)
    if len(x) < (SMA_LONG + 30):
        return pd.DataFrame()

    tick = df["close"].iloc[-1] * 0.0001
    trades = []

    start_idx = SMA_LONG + 20

    for i in range(start_idx, len(x) - 10):
        # Regime filter
        if not (x["trend_up"].iloc[i] and x["slope_up"].iloc[i]):
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

            entry = x["high"].iloc[i] + tick
            stop = x["low"].iloc[i] - tick

            filled = False
            fill_idx = -1

            for w in range(1, max_wait + 1):
                curr = x.iloc[i + w]
                if curr["high"] >= entry:
                    filled = True
                    fill_idx = i + w
                    break

                if setup in ["8.2", "8.3"]:
                    # mantém apenas enquanto slope permanecer ok
                    if not x["slope_up"].iloc[i + w]:
                        break

                    # trailing entry/stop (se preço não bater entrada)
                    if curr["high"] < entry - tick:
                        entry = curr["high"] + tick
                        stop = min(stop, curr["low"] - tick)
                else:
                    break

            if not filled:
                continue

            res_row = {"timeframe": tf, "setup": setup}
            risk = abs(entry - stop)
            if risk <= 0 or np.isnan(risk):
                continue

            for rr in RRS:
                target = entry + (risk * rr)
                outcome = "loss"
                for k in range(fill_idx, min(fill_idx + 50, len(x))):
                    c = x.iloc[k]
                    if c["low"] <= stop:
                        outcome = "loss"
                        break
                    if c["high"] >= target:
                        outcome = "win"
                        break
                res_row[f"rr_{rr}"] = outcome

            trades.append(res_row)

    return pd.DataFrame(trades)

# =============================
# MAIN
# =============================
def main():
    os.makedirs("results", exist_ok=True)

    df_raw = fetch_history(SYMBOL)
    if df_raw.empty:
        print("Erro: Sem dados baixados.")
        pd.DataFrame({"status": ["no_data"]}).to_csv(f"results/baseline_trades_{SYMBOL}.csv", index=False)
        return

    all_trades = []
    for tf in TIMEFRAMES:
        print(f"Rodando {tf}...")
        try:
            df_tf = resample(df_raw, tf)
            if len(df_tf) > 100:
                t = run_backtest(df_tf, tf)
                if not t.empty:
                    all_trades.append(t)
        except Exception as e:
            print(f"Erro em {tf}: {e}")

    if all_trades:
        final = pd.concat(all_trades, ignore_index=True)
        final.to_csv(f"results/baseline_trades_{SYMBOL}.csv", index=False)

        summary = []
        for (tf, st), g in final.groupby(["timeframe", "setup"]):
            row = {"TF": tf, "Setup": st, "Trades": len(g)}
            for rr in RRS:
                w = (g[f"rr_{rr}"] == "win").sum()
                row[f"WR {rr}"] = f"{(w/len(g)):.1%}"
            summary.append(row)

        sum_df = pd.DataFrame(summary).sort_values(["TF", "Setup"])
        print(tabulate(sum_df, headers="keys", tablefmt="grid", showindex=False))

        with open(f"results/baseline_summary_{SYMBOL}.md", "w", encoding="utf-8") as f:
            f.write(f"# Baseline Long Only - {SYMBOL}\n\n")
            f.write(tabulate(sum_df, headers="keys", tablefmt="pipe", showindex=False))

    else:
        print("0 trades gerados.")
        pd.DataFrame({"status": ["no_trades"]}).to_csv(f"results/baseline_trades_{SYMBOL}.csv", index=False)


if __name__ == "__main__":
    main()
    
