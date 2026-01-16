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

# Filtro de inclinação (por padrão usa SMA10 e compara com o candle anterior)
SLOPE_LOOKBACK = int(os.getenv("SLOPE_LOOKBACK", "1"))  # 1 = sma10[i] vs sma10[i-1]

# RRs para simular (removido 3:1)
RRS = [float(x) for x in os.getenv("RRS", "1,1.5,2").split(",")]

# Se no mesmo candle bater TP e SL
AMBIGUOUS_POLICY = os.getenv("AMBIGUOUS_POLICY", "loss").lower()  # loss|win|skip

# Máximo tempo para a ordem "pegar" após o candle do sinal
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
    """
    Formata taxa de acerto como:
      - 43,2%
      - 9,8%
      - 100%
    (no máximo 3 dígitos antes do %)
    """
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

    # tendência base
    x["trend_up"] = (x["close"] > x["sma10"]) & (x["close"] > x["sma100"]) & (x["sma10"] > x["sma100"])
    x["trend_down"] = (x["close"] < x["sma10"]) & 
