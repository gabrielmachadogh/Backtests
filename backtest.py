import os
import time
import requests
import numpy as np
import pandas as pd

BASE_URL = os.getenv("MEXC_CONTRACT_BASE_URL", "https://contract.mexc.com/api/v1")
SYMBOL = os.getenv("SYMBOL", "BTC_USDT")

TIMEFRAMES = [x.strip().lower() for x in os.getenv("TIMEFRAMES", "1h,2h,4h,1d,1w").split(",") if x.strip()]
SETUPS = [x.strip().upper() for x in os.getenv("SETUPS", "PFR,DL").split(",") if x.strip()]

# Médias fixas
SMA_SHORT = int(os.getenv("SMA_SHORT", "8"))
SMA_LONG = int(os.getenv("SMA_LONG", "80"))

# Inclinação: SMA curta acima/abaixo dos últimos N (N=8)
SLOPE_LOOKBACK = int(os.getenv("SLOPE_LOOKBACK", "8"))

# ATR
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))

# RRs
RRS = [float(x) for x in os.getenv("RRS", "1,1.5,2").split(",")]

# Execução
AMBIGUOUS_POLICY = os.getenv("AMBIGUOUS_POLICY", "loss").lower()  # loss|win|skip
MAX_ENTRY_WAIT_BARS = int(os.getenv("MAX_ENTRY_WAIT_BARS", "1"))   # solicitado: 1
MAX_HOLD_BARS = int(os.getenv("MAX_HOLD_BARS", "50"))

# Histórico
MAX_BARS_1H = int(os.getenv("MAX_BARS_1H", "20000"))
WINDOW_DAYS = int(os.getenv("WINDOW_DAYS", "30"))

# Tick size
TICK_SIZE = float(os.getenv("TICK_SIZE", "0"))

# Buckets low para slope_strength
LOW_PCTS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]

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


def add_atr(df: pd.DataFrame, period: int) -> pd.DataFrame:
    x = df.copy()
    prev_close = x["close"].shift(1)
    tr = pd.concat([
        (x["high"] - x["low"]).abs(),
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    x["atr"] = tr.rolling(period).mean()
    return x


# -------------------- indicadores e filtros --------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["sma_s"] = sma(x["close"], SMA_SHORT)
    x["sma_l"] = sma(x["close"], SMA_LONG)

    x["trend_up"] = ((x["close"] > x["sma_s"]) & (x["close"] > x["sma_l"]) & (x["sma_s"] > x["sma_l"]))
    x["trend_down"] = ((x["close"] < x["sma_s"]) & (x["close"] < x["sma_l"]) & (x["sma_s"] < x["sma_l"]))

    prev_max = x["sma_s"].shift(1).rolling(SLOPE_LOOKBACK).max()
    prev_min = x["sma_s"].shift(1).rolling(SLOPE_LOOKBACK).min()

    x["slope_up"] = x["sma_s"] > prev_max
    x["slope_down"] = x["sma_s"] < prev_min

    # sempre positivo (magnitude) para comparar
    x["slope_strength_up"] = x["sma_s"] - prev_max
    x["slope_strength_down"] = prev_min - x["sma_s"]

    x["ma_gap_pct"] = (x["sma_s"] - x["sma_l"]) / x["sma_l"] * 100.0
    x["dist_to_sma80_pct"] = (x["close"] - x["sma_l"]) / x["sma_l"] * 100.0

    x = add_atr(x, ATR_PERIOD)
    x["atr_pct"] = (x["atr"] / x["close"]) * 100.0

    x["range"] = (x["high"] - x["low"])
    x["range_pct"] = (x["range"] / x["close"]) * 100.0

    denom = (x["high"] - x["low"]).replace(0, np.nan)
    x["clv"] = (x["close"] - x["low"]) / denom

    return x


# -------------------- setups (apenas PFR e DL) --------------------
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


def dl_buy_signal(x: pd.DataFrame, i: int) -> bool:
    return (x.loc[i, "low"] < x.loc[i - 1, "low"]) and (x.loc[i, "low"] < x.loc[i - 2, "low"])


def dl_sell_signal(x: pd.DataFrame, i: int) -> bool:
    return (x.loc[i, "high"] > x.loc[i - 1, "high"]) and (x.loc[i, "high"] > x.loc[i - 2, "high"])


# -------------------- execução / fill / TP/SL --------------------
def find_fill_long(x: pd.DataFrame, entry_price: float, start_idx: int):
    # como MAX_ENTRY_WAIT_BARS=1, só confere o próximo candle
    if start_idx >= len(x):
        return None
    return start_idx if float(x.loc[start_idx, "high"]) >= entry_price else None


def find_fill_short(x: pd.DataFrame, entry_price: float, start_idx: int):
    if start_idx >= len(x):
        return None
    return start_idx if float(x.loc[start_idx, "low"]) <= entry_price else None


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
    x = add_indicators(df).reset_index(drop=True)
    tick = TICK_SIZE if TICK_SIZE > 0 else infer_tick_size_from_prices(x["close"])

    start = max(SMA_LONG + SLOPE_LOOKBACK + ATR_PERIOD + 20, 5)
    rows = []

    def add_trade(i: int, setup_name: str, side: str):
        if side == "long":
            entry_price = float(x.loc[i, "high"]) + tick
            stop_price = float(x.loc[i, "low"]) - tick
            fill_idx = find_fill_long(x, entry_price, i + 1)
            slope_strength = float(x.loc[i, "slope_strength_up"]) if pd.notna(x.loc[i, "slope_strength_up"]) else np.nan
        else:
            entry_price = float(x.loc[i, "low"]) - tick
            stop_price = float(x.loc[i, "high"]) + tick
            fill_idx = find_fill_short(x, entry_price, i + 1)
            slope_strength = float(x.loc[i, "slope_strength_down"]) if pd.notna(x.loc[i, "slope_strength_down"]) else np.nan

        if fill_idx is None:
            return

        fill_delay = int(fill_idx - i)  # sempre 1 com este modelo

        row = {
            "timeframe": tf_name,
            "setup": setup_name,
            "side": side,
            "signal_time": x.loc[i, "ts"],
            "entry_time": x.loc[fill_idx, "ts"],
            "fill_delay": fill_delay,
            "entry_price": entry_price,
            "stop_price": stop_price,

            # features do candle do sinal:
            "ma_gap_pct": float(x.loc[i, "ma_gap_pct"]) if pd.notna(x.loc[i, "ma_gap_pct"]) else np.nan,
            "dist_to_sma80_pct": float(x.loc[i, "dist_to_sma80_pct"]) if pd.notna(x.loc[i, "dist_to_sma80_pct"]) else np.nan,
            "atr_pct": float(x.loc[i, "atr_pct"]) if pd.notna(x.loc[i, "atr_pct"]) else np.nan,
            "clv": float(x.loc[i, "clv"]) if pd.notna(x.loc[i, "clv"]) else np.nan,
            "range_pct": float(x.loc[i, "range_pct"]) if pd.notna(x.loc[i, "range_pct"]) else np.nan,
            "slope_strength": slope_strength,
        }

        for rr in RRS:
            row[f"rr_{rr}"] = simulate_tp_sl(x, fill_idx, side, entry_price, stop_price, rr)

        rows.append(row)

    for i in range(start, len(x) - 1):
        if pd.isna(x.loc[i, "sma_s"]) or pd.isna(x.loc[i, "sma_l"]) or pd.isna(x.loc[i, "atr_pct"]):
            continue
        if pd.isna(x.loc[i, "slope_strength_up"]) and pd.isna(x.loc[i, "slope_strength_down"]):
            continue

        trend_up = bool(x.loc[i, "trend_up"])
        trend_down = bool(x.loc[i, "trend_down"])
        slope_up = bool(x.loc[i, "slope_up"])
        slope_down = bool(x.loc[i, "slope_down"])

        if trend_up and slope_up:
            if "PFR" in SETUPS and pfr_buy_signal(x, i):
                add_trade(i, "PFR", "long")
            if "DL" in SETUPS and dl_buy_signal(x, i):
                add_trade(i, "DL", "long")

        if trend_down and slope_down:
            if "PFR" in SETUPS and pfr_sell_signal(x, i):
                add_trade(i, "PFR", "short")
            if "DL" in SETUPS and dl_sell_signal(x, i):
                add_trade(i, "DL", "short")

    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame, slope_filter: str = "ALL") -> pd.DataFrame:
    cols = ["slope_filter", "timeframe", "setup", "rr", "trades", "wins", "losses", "no_hit", "skipped", "win_rate", "win_rate_pct"]
    if trades.empty:
        return pd.DataFrame(columns=cols)

    summaries = []
    for tf in sorted(trades["timeframe"].unique()):
        rtf = trades[trades["timeframe"] == tf]
        for setup in sorted(rtf["setup"].unique()):
            rs = rtf[rtf["setup"] == setup]
            for rr in RRS:
                col = f"rr_{rr}"
                wins = int((rs[col] == "win").sum())
                losses = int((rs[col] == "loss").sum())
                no_hit = int((rs[col] == "no_hit").sum())
                skipped = int((rs[col] == "skip").sum())
                trades_n = wins + losses + no_hit + skipped

                denom = wins + losses
                win_rate = (wins / denom) if denom > 0 else np.nan

                summaries.append({
                    "slope_filter": slope_filter,
                    "timeframe": tf,
                    "setup": setup,
                    "rr": rr,
                    "trades": trades_n,
                    "wins": wins,
                    "losses": losses,
                    "no_hit": no_hit,
                    "skipped": skipped,
                    "win_rate": win_rate,
                    "win_rate_pct": fmt_pct(win_rate),
                })

    return pd.DataFrame(summaries, columns=cols)


def apply_slope_low_filter(trades: pd.DataFrame, p: float) -> pd.DataFrame:
    """
    Mantém apenas trades no LOW p% de slope_strength, calculando quantil por (timeframe, setup).
    """
    if trades.empty:
        return trades

    out_parts = []
    for (tf, setup), g in trades.groupby(["timeframe", "setup"], dropna=False):
        g2 = g.dropna(subset=["slope_strength"]).copy()
        if g2.empty:
            continue
        q = g2["slope_strength"].quantile(p)
        out_parts.append(g2[g2["slope_strength"] <= q])

    if not out_parts:
        return pd.DataFrame(columns=trades.columns)

    return pd.concat(out_parts, ignore_index=True)


def save_markdown_table(df: pd.DataFrame, title: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if df.empty:
            f.write("Sem dados.\n")
            return
        f.write("| SlopeFilter | TF | Setup | RR | Trades | Wins | Losses | WinRate |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|\n")
        for _, r in df.iterrows():
            f.write(
                f"| {r['slope_filter']} | {r['timeframe']} | {r['setup']} | {r['rr']} | {int(r['trades'])} | {int(r['wins'])} | {int(r['losses'])} | {r.get('win_rate_pct','-')} |\n"
            )


def main():
    os.makedirs("results", exist_ok=True)

    if DEBUG:
        print("[debug] CONFIG:",
              SYMBOL, TIMEFRAMES, SETUPS, SMA_SHORT, SMA_LONG, SLOPE_LOOKBACK,
              ATR_PERIOD, RRS, MAX_ENTRY_WAIT_BARS, MAX_HOLD_BARS)

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

    all_trades = []
    for tf in ["1h", "2h", "4h", "1d", "1w"]:
        if tf not in TIMEFRAMES:
            continue
        df_tf = tf_map.get(tf)
        if df_tf is None or df_tf.empty:
            continue
        min_len = SMA_LONG + SLOPE_LOOKBACK + ATR_PERIOD + 50
        if len(df_tf) < min_len:
            continue
        all_trades.append(backtest_setups(df_tf, tf))

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    # 1) salvar trades SEMPRE
    trades_path = f"results/backtest_trades_{SYMBOL}.csv"
    trades_df.to_csv(trades_path, index=False)

    # 2) summary ALL + slope low buckets
    summaries = []
    summaries.append(summarize(trades_df, slope_filter="ALL"))

    for p in LOW_PCTS:
        label = f"low{int(p*100)}"
        fdf = apply_slope_low_filter(trades_df, p)
        summaries.append(summarize(fdf, slope_filter=label))

    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

    summary_df.to_csv(f"results/backtest_summary_{SYMBOL}.csv", index=False)
    save_markdown_table(summary_df, f"Summary (ALL + slope low buckets) — {SYMBOL}", f"results/backtest_summary_{SYMBOL}.md")


if __name__ == "__main__":
    main()
