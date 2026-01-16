import os
import time
import requests
import numpy as np
import pandas as pd

BASE_URL = os.getenv("MEXC_CONTRACT_BASE_URL", "https://contract.mexc.com/api/v1")

SYMBOL = os.getenv("SYMBOL", "BTC_USDT")

# Timeframes desejados
TIMEFRAMES = [x.strip().lower() for x in os.getenv("TIMEFRAMES", "1h,2h,4h,1d,1w").split(",") if x.strip()]

# Quais setups rodar (PFR, DL, IB)
SETUPS = [x.strip().upper() for x in os.getenv("SETUPS", "PFR,DL,IB").split(",") if x.strip()]

# Trend filter (SMA 10/100)
SMA_SHORT = int(os.getenv("SMA_SHORT", "10"))
SMA_LONG = int(os.getenv("SMA_LONG", "100"))

# Inclinação: média da variação da SMA10 nos últimos N candles (default 5)
SLOPE_PERIOD = int(os.getenv("SLOPE_PERIOD", "5"))

# RRs para simular (sem 3:1)
RRS = [float(x) for x in os.getenv("RRS", "1,1.5,2").split(",")]

# Se no mesmo candle bater TP e SL
AMBIGUOUS_POLICY = os.getenv("AMBIGUOUS_POLICY", "loss").lower()  # loss|win|skip

# Máximo tempo para a ordem "pegar" após o candle do sinal (máx 3 candles)
MAX_ENTRY_WAIT_BARS = int(os.getenv("MAX_ENTRY_WAIT_BARS", "3"))

# Máximo tempo para bater TP/SL depois de entrar
MAX_HOLD_BARS = int(os.getenv("MAX_HOLD_BARS", "50"))

# Quanto histórico 1h tentar puxar
MAX_BARS_1H = int(os.getenv("MAX_BARS_1H", "20000"))
WINDOW_DAYS = int(os.getenv("WINDOW_DAYS", "30"))

# Tick size (se 0, ele infere pelo número de casas do preço)
TICK_SIZE = float(os.getenv("TICK_SIZE", "0"))

DEBUG = os.getenv("DEBUG", "0") == "1"


def fmt_pct(win_rate) -> str:
    """Formata taxa de acerto como 43,2% (máx 3 dígitos antes do %)."""
    try:
        if win_rate is None:
            return "-"
        if isinstance(win_rate, (float, np.floating)) and np.isnan(win_rate):
            return "-"
        pct = float(win_rate) * 100.0
    except Exception:
        return "-"

    if pct >= 100:
        return "100%"

    s = f"{pct:.1f}".replace(".", ",")
    if s.endswith(",0"):
        s = s[:-2]
    return f"{s}%"


# -------------------- infra / parsing --------------------
def http_get_json(url, params=None, tries=3, timeout=25):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def to_datetime_auto(ts_series: pd.Series) -> pd.Series:
    s = pd.to_numeric(ts_series, errors="coerce")
    unit = "s" if s.dropna().median() < 1e12 else "ms"
    return pd.to_datetime(s, unit=unit, utc=True)


def parse_kline_to_df(payload):
    if isinstance(payload, dict):
        data = payload.get("data") or payload.get("datas") or payload.get("result")
    else:
        data = payload

    if data is None:
        raise RuntimeError(f"Resposta sem data: {payload}")

    if isinstance(data, dict) and "time" in data:
        df = pd.DataFrame({
            "ts": data["time"],
            "open": data.get("open"),
            "high": data.get("high"),
            "low": data.get("low"),
            "close": data.get("close"),
            "volume": data.get("vol") or data.get("volume"),
        })
    elif isinstance(data, list):
        if len(data) == 0:
            return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        first = data[0]
        if isinstance(first, (list, tuple)):
            df = pd.DataFrame(data).iloc[:, :6]
            df.columns = ["ts", "open", "high", "low", "close", "volume"]
        elif isinstance(first, dict):
            df = pd.DataFrame(data)
            rename = {}
            for a, b in [
                ("time", "ts"), ("timestamp", "ts"), ("t", "ts"),
                ("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"),
                ("v", "volume"), ("vol", "volume")
            ]:
                if a in df.columns and b not in df.columns:
                    rename[a] = b
            df = df.rename(columns=rename)
            df = df[["ts", "open", "high", "low", "close", "volume"]]
        else:
            raise RuntimeError(f"Formato de kline inesperado: {first}")
    else:
        raise RuntimeError(f"Formato de kline inesperado: {type(data)}")

    df["ts"] = to_datetime_auto(df["ts"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts").reset_index(drop=True)
    return df


def fetch_ohlcv_1h_max(symbol: str, max_bars: int, window_days: int):
    interval = "Min60"
    now_s = int(time.time())
    end_s = now_s

    step_s = window_days * 24 * 60 * 60
    dfs = []

    while True:
        start_s = end_s - step_s
        url = f"{BASE_URL}/contract/kline/{symbol}"
        params = {"interval": interval, "start": start_s, "end": end_s}
        payload = http_get_json(url, params=params)
        df = parse_kline_to_df(payload)

        if df.empty:
            break

        dfs.append(df)

        oldest = int(df["ts"].iloc[0].timestamp())
        if oldest >= end_s:
            break
        end_s = oldest - 1

        total = sum(len(x) for x in dfs)
        if total >= max_bars:
            break

        time.sleep(0.2)

    if not dfs:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    out = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if len(out) > max_bars:
        out = out.tail(max_bars).reset_index(drop=True)
    return out


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    x = df.copy().set_index("ts")
    y = x.resample(rule).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open", "high", "low", "close"]).reset_index()
    return y


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def infer_tick_size_from_prices(close_series: pd.Series) -> float:
    last = float(close_series.dropna().iloc[-1])
    s = f"{last:.10f}".rstrip("0").rstrip(".")
    if "." in s:
        dec = len(s.split(".")[1])
        return 10 ** (-dec)
    return 1.0


# -------------------- tendência + filtros --------------------
def add_trend_columns(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["sma10"] = sma(x["close"], SMA_SHORT)
    x["sma100"] = sma(x["close"], SMA_LONG)

    x["trend_up"] = (
        (x["close"] > x["sma10"])
        & (x["close"] > x["sma100"])
        & (x["sma10"] > x["sma100"])
    )

    x["trend_down"] = (
        (x["close"] < x["sma10"])
        & (x["close"] < x["sma100"])
        & (x["sma10"] < x["sma100"])
    )

    # Inclinação baseada nos últimos SLOPE_PERIOD candles:
    # slope = média dos diffs recentes da SMA10
    x["sma10_slope"] = x["sma10"].diff().rolling(SLOPE_PERIOD).mean()
    x["sma10_slope_up"] = x["sma10_slope"] > 0
    x["sma10_slope_down"] = x["sma10_slope"] < 0

    return x


# -------------------- setups --------------------
# PFR
def pfr_buy_signal(x: pd.DataFrame, i: int) -> bool:
    return (
        (x.loc[i, "low"] < x.loc[i - 1, "low"])
        and (x.loc[i, "low"] < x.loc[i - 2, "low"])
        and (x.loc[i, "close"] > x.loc[i - 1, "close"])
    )


def pfr_sell_signal(x: pd.DataFrame, i: int) -> bool:
    return (
        (x.loc[i, "high"] > x.loc[i - 1, "high"])
        and (x.loc[i, "high"] > x.loc[i - 2, "high"])
        and (x.loc[i, "close"] < x.loc[i - 1, "close"])
    )


# Dave Landry
def dl_buy_signal(x: pd.DataFrame, i: int) -> bool:
    return (x.loc[i, "low"] < x.loc[i - 1, "low"]) and (x.loc[i, "low"] < x.loc[i - 2, "low"])


def dl_sell_signal(x: pd.DataFrame, i: int) -> bool:
    return (x.loc[i, "high"] > x.loc[i - 1, "high"]) and (x.loc[i, "high"] > x.loc[i - 2, "high"])


# Inside Bar
def ib_signal(x: pd.DataFrame, i: int) -> bool:
    return (x.loc[i, "high"] <= x.loc[i - 1, "high"]) and (x.loc[i, "low"] >= x.loc[i - 1, "low"])


# -------------------- execução/TP/SL --------------------
def find_fill_long(x: pd.DataFrame, entry_price: float, start_idx: int, max_wait: int):
    end = min(start_idx + max_wait, len(x))
    for j in range(start_idx, end):
        if float(x.loc[j, "high"]) >= entry_price:
            return j
    return None


def find_fill_short(x: pd.DataFrame, entry_price: float, start_idx: int, max_wait: int):
    end = min(start_idx + max_wait, len(x))
    for j in range(start_idx, end):
        if float(x.loc[j, "low"]) <= entry_price:
            return j
    return None


def simulate_tp_sl(x: pd.DataFrame, entry_idx: int, side: str, entry_price: float, stop_price: float, rr: float):
    end_j = min(entry_idx + MAX_HOLD_BARS, len(x))

    if side == "long":
        risk = entry_price - stop_price
        if risk <= 0:
            return "skip"
        tp = entry_price + rr * risk
        sl = stop_price

        for j in range(entry_idx, end_j):
            h = float(x.loc[j, "high"])
            l = float(x.loc[j, "low"])
            hit_tp = h >= tp
            hit_sl = l <= sl

            if hit_tp and hit_sl:
                if AMBIGUOUS_POLICY == "skip":
                    return "skip"
                return "win" if AMBIGUOUS_POLICY == "win" else "loss"
            if hit_sl:
                return "loss"
            if hit_tp:
                return "win"
        return "no_hit"

    if side == "short":
        risk = stop_price - entry_price
        if risk <= 0:
            return "skip"
        tp = entry_price - rr * risk
        sl = stop_price

        for j in range(entry_idx, end_j):
            h = float(x.loc[j, "high"])
            l = float(x.loc[j, "low"])
            hit_tp = l <= tp
            hit_sl = h >= sl

            if hit_tp and hit_sl:
                if AMBIGUOUS_POLICY == "skip":
                    return "skip"
                return "win" if AMBIGUOUS_POLICY == "win" else "loss"
            if hit_sl:
                return "loss"
            if hit_tp:
                return "win"
        return "no_hit"

    return "skip"


def backtest_setups(df: pd.DataFrame, tf_name: str) -> pd.DataFrame:
    x = add_trend_columns(df).reset_index(drop=True)
    tick = TICK_SIZE if TICK_SIZE > 0 else infer_tick_size_from_prices(x["close"])

    rows = []
    start = max(SMA_LONG + 2, 2)

    def add_trade(i: int, setup_name: str, side: str):
        if side == "long":
            entry_price = float(x.loc[i, "high"]) + tick
            stop_price = float(x.loc[i, "low"]) - tick
            fill_idx = find_fill_long(x, entry_price, i + 1, MAX_ENTRY_WAIT_BARS)
        else:
            entry_price = float(x.loc[i, "low"]) - tick
            stop_price = float(x.loc[i, "high"]) + tick
            fill_idx = find_fill_short(x, entry_price, i + 1, MAX_ENTRY_WAIT_BARS)

        if fill_idx is None:
            return

        row = {
            "timeframe": tf_name,
            "setup": setup_name,
            "signal_time": x.loc[i, "ts"],
            "entry_time": x.loc[fill_idx, "ts"],
            "side": side,
            "entry_price": entry_price,
            "stop_price": stop_price,
        }
        for rr in RRS:
            row[f"rr_{rr}"] = simulate_tp_sl(x, fill_idx, side, entry_price, stop_price, rr)
        rows.append(row)

    for i in range(start, len(x) - 1):
        if pd.isna(x.loc[i, "sma10"]) or pd.isna(x.loc[i, "sma100"]):
            continue

        # precisa ter slope calculado
        if pd.isna(x.loc[i, "sma10_slope"]):
            continue

        trend_up = bool(x.loc[i, "trend_up"])
        trend_down = bool(x.loc[i, "trend_down"])

        slope_up = bool(x.loc[i, "sma10_slope_up"])
        slope_down = bool(x.loc[i, "sma10_slope_down"])

        # Compra: tendência alta + inclinação positiva
        if trend_up and slope_up:
            if "PFR" in SETUPS and pfr_buy_signal(x, i):
                add_trade(i, "PFR", "long")
            if "DL" in SETUPS and dl_buy_signal(x, i):
                add_trade(i, "DL", "long")
            if "IB" in SETUPS and ib_signal(x, i):
                add_trade(i, "IB", "long")

        # Venda: tendência baixa + inclinação negativa
        if trend_down and slope_down:
            if "PFR" in SETUPS and pfr_sell_signal(x, i):
                add_trade(i, "PFR", "short")
            if "DL" in SETUPS and dl_sell_signal(x, i):
                add_trade(i, "DL", "short")
            if "IB" in SETUPS and ib_signal(x, i):
                add_trade(i, "IB", "short")

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    cols = ["timeframe", "setup", "rr", "trades", "wins", "losses", "no_hit", "skipped", "win_rate", "win_rate_pct"]
    if results.empty:
        return pd.DataFrame(columns=cols)

    summaries = []
    for tf in sorted(results["timeframe"].unique()):
        rtf = results[results["timeframe"] == tf]
        for setup in sorted(rtf["setup"].unique()):
            rs = rtf[rtf["setup"] == setup]
            for rr in RRS:
                col = f"rr_{rr}"
                wins = int((rs[col] == "win").sum())
                losses = int((rs[col] == "loss").sum())
                no_hit = int((rs[col] == "no_hit").sum())
                skipped = int((rs[col] == "skip").sum())
                trades = wins + losses + no_hit + skipped

                denom = wins + losses
                win_rate = (wins / denom) if denom > 0 else np.nan

                summaries.append({
                    "timeframe": tf,
                    "setup": setup,
                    "rr": rr,
                    "trades": trades,
                    "wins": wins,
                    "losses": losses,
                    "no_hit": no_hit,
                    "skipped": skipped,
                    "win_rate": win_rate,
                    "win_rate_pct": fmt_pct(win_rate),
                })

    return pd.DataFrame(summaries, columns=cols)


def save_markdown(summary_df: pd.DataFrame, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Backtest (MEXC Perps) — {SYMBOL}\n\n")
        f.write(f"- Gerado em UTC: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}\n")
        f.write(f"- Timeframes: `{', '.join(TIMEFRAMES)}`\n")
        f.write(f"- Tendência: SMA{SMA_SHORT}/SMA{SMA_LONG} (close acima/abaixo das duas + cruzamento)\n")
        f.write(f"- Filtro inclinação: média da variação da SMA{SMA_SHORT} nos últimos {SLOPE_PERIOD} candles (buy>0, sell<0)\n")
        f.write(f"- Setups: `{', '.join(SETUPS)}`\n")
        f.write(f"- RRs: `{', '.join([str(r) for r in RRS])}`\n")
        f.write(f"- MAX_ENTRY_WAIT_BARS: `{MAX_ENTRY_WAIT_BARS}` | MAX_HOLD_BARS: `{MAX_HOLD_BARS}` | AMBIGUOUS_POLICY: `{AMBIGUOUS_POLICY}`\n\n")

        if summary_df.empty:
            f.write("Sem resultados.\n")
            return

        f.write("| TF | Setup | RR | Trades | Wins | Losses | NoHit | Skipped | WinRate |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, r in summary_df.iterrows():
            f.write(
                f"| {r['timeframe']} | {r['setup']} | {r['rr']} | {int(r['trades'])} | {int(r['wins'])} | {int(r['losses'])} | {int(r['no_hit'])} | {int(r['skipped'])} | {r.get('win_rate_pct','-')} |\n"
            )


def main():
    os.makedirs("results", exist_ok=True)

    df_1h = fetch_ohlcv_1h_max(SYMBOL, max_bars=MAX_BARS_1H, window_days=WINDOW_DAYS)
    if df_1h.empty:
        raise RuntimeError("Não consegui baixar candles 1h.")

    tf_map = {"1h": df_1h}
    if "2h" in TIMEFRAMES:
        tf_map["2h"] = resample_ohlcv(df_1h, "2H")
    if "4h" in TIMEFRAMES:
        tf_map["4h"] = resample_ohlcv(df_1h, "4H")
    if "1d" in TIMEFRAMES:
        tf_map["1d"] = resample_ohlcv(df_1h, "1D")
    if "1w" in TIMEFRAMES:
        tf_map["1w"] = resample_ohlcv(df_1h, "W-SUN")

    all_results = []
    for tf in ["1h", "2h", "4h", "1d", "1w"]:
        if tf not in TIMEFRAMES:
            continue
        df = tf_map.get(tf)
        if df is None or df.empty:
            continue
        if len(df) < (SMA_LONG + SLOPE_PERIOD + 10):
            continue
        all_results.append(backtest_setups(df, tf))

    trades_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    summary_df = summarize(trades_df)

    trades_df.to_csv(f"results/backtest_trades_{SYMBOL}.csv", index=False)
    summary_df.to_csv(f"results/backtest_summary_{SYMBOL}.csv", index=False)
    save_markdown(summary_df, f"results/backtest_summary_{SYMBOL}.md")


if __name__ == "__main__":
    main()
