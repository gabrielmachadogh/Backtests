import os
import time
import requests
import numpy as np
import pandas as pd

BASE_URL = os.getenv("MEXC_CONTRACT_BASE_URL", "https://contract.mexc.com/api/v1")

SYMBOL = os.getenv("SYMBOL", "BTC_USDT")

# Timeframes desejados
TIMEFRAMES = [x.strip().lower() for x in os.getenv("TIMEFRAMES", "1h,2h,4h,1d,1w").split(",") if x.strip()]

# Setup/indicadores
SHORT_MA = int(os.getenv("SHORT_MA", "10"))
LONG_MA = int(os.getenv("LONG_MA", "100"))
MA_TYPE = os.getenv("MA_TYPE", "sma").lower()  # sma|ema

ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULT = float(os.getenv("ATR_MULT", "1.0"))  # define distância do stop
MAX_HOLD_BARS = int(os.getenv("MAX_HOLD_BARS", "50"))  # se não bater TP nem SL até aqui => "no_hit"

# RRs para simular
RRS = [float(x) for x in os.getenv("RRS", "1,1.5,2,3").split(",")]

# Se no mesmo candle bater TP e SL, qual regra usar?
# "loss" (conservador), "win" (otimista), "skip" (ignora o trade)
AMBIGUOUS_POLICY = os.getenv("AMBIGUOUS_POLICY", "loss").lower()

DEBUG = os.getenv("DEBUG", "0") == "1"


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


def calc_ma(series: pd.Series, period: int, ma_type: str) -> pd.Series:
    if ma_type == "sma":
        return series.rolling(period).mean()
    if ma_type == "ema":
        return series.ewm(span=period, adjust=False).mean()
    raise ValueError("MA_TYPE deve ser 'sma' ou 'ema'")


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

    df = df.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts")
    return df


def fetch_ohlcv_1h_max(symbol: str, max_bars: int = 20000, window_days: int = 30):
    """
    Baixa o máximo que der de 1h, paginando por janelas de tempo.
    """
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

    out = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["ts"]).sort_values("ts")
    if len(out) > max_bars:
        out = out.tail(max_bars)
    return out


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    rule exemplos: '2H', '4H', '1D', 'W-SUN'
    """
    x = df.copy()
    x = x.set_index("ts")
    y = x.resample(rule).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open", "high", "low", "close"]).reset_index()
    return y


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


# ========= SETUP (VOCÊ VAI TROCAR AQUI DEPOIS) =========
def generate_entries_default(df: pd.DataFrame):
    """
    Setup padrão: filtro de tendência MA + preço acima/abaixo
    Long quando: ma_curta>ma_longa e close>ambas
    Short quando: ma_curta<ma_longa e close<ambas
    Entrada: OPEN do próximo candle quando o sinal "vira" True
    Stop: ATR_MULT * ATR (atr do candle do sinal)
    """
    x = df.copy()
    x["ma_s"] = calc_ma(x["close"], SHORT_MA, MA_TYPE)
    x["ma_l"] = calc_ma(x["close"], LONG_MA, MA_TYPE)

    long_cond = (x["ma_s"] > x["ma_l"]) & (x["close"] > x["ma_s"]) & (x["close"] > x["ma_l"])
    short_cond = (x["ma_s"] < x["ma_l"]) & (x["close"] < x["ma_s"]) & (x["close"] < x["ma_l"])

    long_entry = long_cond & (~long_cond.shift(1).fillna(False))
    short_entry = short_cond & (~short_cond.shift(1).fillna(False))

    entries = []
    for i in range(len(x) - 1):  # precisa ter próximo candle
        if pd.isna(x.loc[i, "atr"]):
            continue

        risk = float(x.loc[i, "atr"]) * ATR_MULT
        if risk <= 0:
            continue

        if bool(long_entry.iloc[i]):
            entry_idx = i + 1
            entry_price = float(x.loc[entry_idx, "open"])
            stop_price = entry_price - risk
            entries.append((entry_idx, "long", entry_price, stop_price))

        if bool(short_entry.iloc[i]):
            entry_idx = i + 1
            entry_price = float(x.loc[entry_idx, "open"])
            stop_price = entry_price + risk
            entries.append((entry_idx, "short", entry_price, stop_price))

    return entries
# ======================================================


def simulate_trade(df: pd.DataFrame, entry_idx: int, side: str, entry_price: float, stop_price: float, rr: float):
    """
    Retorna: "win" | "loss" | "no_hit" | "skip"
    """
    end_j = min(entry_idx + MAX_HOLD_BARS, len(df))

    if side == "long":
        tp = entry_price + rr * (entry_price - stop_price)
        sl = stop_price
        for j in range(entry_idx, end_j):
            h = float(df.loc[j, "high"])
            l = float(df.loc[j, "low"])
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
        tp = entry_price - rr * (stop_price - entry_price)
        sl = stop_price
        for j in range(entry_idx, end_j):
            h = float(df.loc[j, "high"])
            l = float(df.loc[j, "low"])
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


def backtest_for_df(df: pd.DataFrame, tf_name: str) -> pd.DataFrame:
    x = add_atr(df, ATR_PERIOD).reset_index(drop=True)

    entries = generate_entries_default(x)  # <- você vai trocar pelos seus setups
    if DEBUG:
        print(f"[debug] {tf_name}: entries={len(entries)} bars={len(x)}")

    rows = []
    for entry_idx, side, entry_price, stop_price in entries:
        row = {
            "timeframe": tf_name,
            "entry_time": x.loc[entry_idx, "ts"],
            "side": side,
            "entry_price": entry_price,
            "stop_price": stop_price,
        }
        for rr in RRS:
            row[f"rr_{rr}"] = simulate_trade(x, entry_idx, side, entry_price, stop_price, rr)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["timeframe", "rr", "trades", "wins", "losses", "no_hit", "skipped", "win_rate"])

    summaries = []
    for tf in sorted(results["timeframe"].unique()):
        rtf = results[results["timeframe"] == tf]
        for rr in RRS:
            col = f"rr_{rr}"
            wins = int((rtf[col] == "win").sum())
            losses = int((rtf[col] == "loss").sum())
            no_hit = int((rtf[col] == "no_hit").sum())
            skipped = int((rtf[col] == "skip").sum())
            trades = wins + losses + no_hit + skipped
            denom = wins + losses  # taxa de acerto considerando só resolvidos
            win_rate = (wins / denom) if denom > 0 else np.nan

            summaries.append({
                "timeframe": tf,
                "rr": rr,
                "trades": trades,
                "wins": wins,
                "losses": losses,
                "no_hit": no_hit,
                "skipped": skipped,
                "win_rate": win_rate,
            })

    return pd.DataFrame(summaries)


def save_markdown(summary_df: pd.DataFrame, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Backtest (MEXC Perps) — {SYMBOL}\n\n")
        f.write(f"- Gerado em UTC: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}\n")
        f.write(f"- Setup (padrão): filtro tendência MA + preço acima/abaixo\n")
        f.write(f"- MA: `{MA_TYPE} {SHORT_MA}/{LONG_MA}` | ATR: `{ATR_PERIOD}` | ATR_MULT: `{ATR_MULT}` | MAX_HOLD_BARS: `{MAX_HOLD_BARS}`\n")
        f.write(f"- Ambiguous (TP e SL no mesmo candle): `{AMBIGUOUS_POLICY}`\n\n")

        if summary_df.empty:
            f.write("Sem resultados.\n")
            return

        f.write("| TF | RR | Trades | Wins | Losses | NoHit | Skipped | WinRate (wins/(wins+losses)) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, r in summary_df.iterrows():
            wr = r["win_rate"]
            wr_str = f"{wr:.3f}" if isinstance(wr, (float, np.floating)) and not np.isnan(wr) else "-"
            f.write(
                f"| {r['timeframe']} | {r['rr']} | {int(r['trades'])} | {int(r['wins'])} | {int(r['losses'])} | {int(r['no_hit'])} | {int(r['skipped'])} | {wr_str} |\n"
            )


def main():
    os.makedirs("results", exist_ok=True)

    print(f"[info] SYMBOL={SYMBOL} TIMEFRAMES={TIMEFRAMES}")
    print(f"[info] MA={MA_TYPE} {SHORT_MA}/{LONG_MA} | ATR={ATR_PERIOD} mult={ATR_MULT} | RRs={RRS} | MAX_HOLD_BARS={MAX_HOLD_BARS}")

    # 1) baixa 1h máximo
    df_1h = fetch_ohlcv_1h_max(SYMBOL, max_bars=20000, window_days=30)
    if df_1h.empty:
        raise RuntimeError("Não consegui baixar candles 1h.")

    # 2) monta dataframes por TF (a partir do 1h)
    tf_map = {"1h": df_1h}
    if "2h" in TIMEFRAMES:
        tf_map["2h"] = resample_ohlcv(df_1h, "2H")
    if "4h" in TIMEFRAMES:
        tf_map["4h"] = resample_ohlcv(df_1h, "4H")
    if "1d" in TIMEFRAMES:
        tf_map["1d"] = resample_ohlcv(df_1h, "1D")
    if "1w" in TIMEFRAMES:
        # semana fechando domingo (UTC)
        tf_map["1w"] = resample_ohlcv(df_1h, "W-SUN")

    # 3) backtest
    all_results = []
    for tf in ["1h", "2h", "4h", "1d", "1w"]:
        if tf not in TIMEFRAMES:
            continue
        df = tf_map.get(tf)
        if df is None or df.empty:
            print(f"[warn] TF={tf}: sem dados. Pulando.")
            continue
        if len(df) < (LONG_MA + ATR_PERIOD + 10):
            print(f"[warn] TF={tf}: pouco histórico ({len(df)}). Pulando.")
            continue
        res = backtest_for_df(df, tf)
        all_results.append(res)

    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    summary_df = summarize(results_df)

    results_df.to_csv(f"results/backtest_trades_{SYMBOL}.csv", index=False)
    summary_df.to_csv(f"results/backtest_summary_{SYMBOL}.csv", index=False)
    save_markdown(summary_df, f"results/backtest_summary_{SYMBOL}.md")

    print("[info] Salvo em results/")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
