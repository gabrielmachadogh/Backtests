import os
import time
import requests
import numpy as np
import pandas as pd
from tabulate import tabulate

# =============================
# CONFIG (env-friendly)
# =============================
BASE_URL = os.getenv("MEXC_CONTRACT_BASE_URL", "https://contract.mexc.com/api/v1")
SYMBOL = os.getenv("SYMBOL", "BTC_USDT")
DEBUG = os.getenv("DEBUG", "0") == "1"

TIMEFRAMES = ["1h", "2h", "4h", "1d", "1w"]

SMA_SHORT = int(os.getenv("SMA_SHORT", "8"))
SMA_LONG = int(os.getenv("SMA_LONG", "80"))
SLOPE_LOOKBACK = int(os.getenv("SLOPE_LOOKBACK", "8"))

RRS = [float(x) for x in os.getenv("RRS", "1.0,1.5,2.0").split(",")]

MAX_BARS_FETCH = int(os.getenv("MAX_BARS_FETCH", "18000"))
WINDOW_DAYS = int(os.getenv("WINDOW_DAYS", "30"))

SAVE_OHLCV = os.getenv("SAVE_OHLCV", "1") == "1"

# ----- ATR -----
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_RISK_MIN = float(os.getenv("ATR_RISK_MIN", "0.6"))  # risk/ATR mínimo
ATR_RISK_MAX = float(os.getenv("ATR_RISK_MAX", "3.0"))  # risk/ATR máximo

# ----- Stretch (preço esticado) -----
# distância máxima do preço em relação à SMA_LONG (SMA80), em %
MAX_DIST_SMA_L_PCT = float(os.getenv("MAX_DIST_SMA_L_PCT", "0.06"))  # 6%

# ----- HTF alignment -----
HTF_REQUIRE_SLOPE = os.getenv("HTF_REQUIRE_SLOPE", "1") == "1"
HTF_MAP = {
    "1h": "4h",
    "2h": "4h",
    "4h": "1d",
    "1d": "1w",
    "1w": None,
}

# ----- Exits -----
TIME_EXIT_BARS = int(os.getenv("TIME_EXIT_BARS", "50"))  # N candles
BE_TRIGGER_R = float(os.getenv("BE_TRIGGER_R", "0.8"))   # +0.8R move stop p/ entrada

# ----- Experimentos (A/B) -----
# Você pode controlar quais experimentos roda via env:
# EXPERIMENTS="base,atr,stretch,htf,timeexit,breakeven,all"
EXPERIMENTS = [x.strip() for x in os.getenv(
    "EXPERIMENTS",
    "base,atr,stretch,htf,timeexit,breakeven,all"
).split(",") if x.strip()]


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


def fetch_history(symbol):
    print(f"Baixando histórico 1h para {symbol}...")
    all_dfs = []
    end_ts = int(time.time())
    step = WINDOW_DAYS * 86400

    total = 0
    for n in range(80):  # limite de segurança
        if total >= MAX_BARS_FETCH:
            break

        start_ts = end_ts - step
        url = f"{BASE_URL}/contract/kline/{symbol}"
        params = {"interval": "Min60", "start": start_ts, "end": end_ts}
        payload = http_get_json(url, params=params)

        if DEBUG and n == 0 and isinstance(payload, dict):
            print("Exemplo payload keys:", list(payload.keys()))

        df = parse_kline(payload)
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
    print(f"Total baixado: {len(full_df)} candles.")
    return full_df


def resample(df, rule):
    if df.empty:
        return df
    if rule == "1h":
        return df.copy()

    # "1W" fecha por padrão no DOMINGO (W-SUN). Se quiser segunda, use "W-MON".
    mapping = {"2h": "2H", "4h": "4H", "1d": "1D", "1w": "1W"}
    if rule not in mapping:
        return df

    dfi = df.set_index("ts")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    res = dfi.resample(mapping[rule]).agg(agg).dropna()
    return res.reset_index()


# =============================
# INDICATORS
# =============================
def add_atr(df, period=14):
    x = df.copy()
    prev_close = x["close"].shift(1)
    tr = pd.concat([
        (x["high"] - x["low"]),
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    x["atr"] = tr.rolling(period).mean()
    return x


def add_indicators(df):
    x = df.copy()
    x["sma_s"] = x["close"].rolling(SMA_SHORT).mean()
    x["sma_l"] = x["close"].rolling(SMA_LONG).mean()

    # Regime (long only)
    x["trend_up"] = (x["close"] > x["sma_s"]) & (x["close"] > x["sma_l"]) & (x["sma_s"] > x["sma_l"])
    x["slope_up"] = x["sma_s"] > x["sma_s"].shift(1).rolling(SLOPE_LOOKBACK).max()

    # Distâncias (%)
    x["dist_sma_l"] = (x["close"] - x["sma_l"]) / x["close"]

    # ATR
    x = add_atr(x, ATR_PERIOD)

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
    Simula a partir do candle fill_idx inclusive.
    Retorna:
      hit_target (bool), exit_type (str), r_result (float), exit_ts (Timestamp)
    """
    risk = entry - initial_stop
    if risk <= 0 or np.isnan(risk):
        return False, "invalid", np.nan, pd.NaT

    stop = initial_stop
    target = entry + (risk * rr)
    be_trigger = entry + (risk * BE_TRIGGER_R)

    be_moved = False
    last_idx = min(fill_idx + TIME_EXIT_BARS - 1, len(x) - 1)

    for k in range(fill_idx, last_idx + 1):
        c = x.iloc[k]

        # Assunção conservadora intrabar:
        # 1) stop
        # 2) target
        # 3) break-even trigger (move stop)
        if c["low"] <= stop:
            r = (stop - entry) / risk  # -1.0 ou 0.0 (se stop virou entry)
            return False, "stop", float(r), c["ts"]

        if c["high"] >= target:
            return True, "target", float(rr), c["ts"]

        if cfg["use_breakeven"] and (not be_moved) and (c["high"] >= be_trigger):
            stop = entry
            be_moved = True

    # Não bateu stop/target dentro de N candles
    if cfg["use_time_exit"]:
        exit_px = float(x["close"].iloc[last_idx])
        r = (exit_px - entry) / risk
        return False, "time_exit", float(r), x["ts"].iloc[last_idx]

    # comportamento antigo: vira loss
    return False, "timeout_loss", -1.0, x["ts"].iloc[last_idx]


# =============================
# BACKTEST CORE
# =============================
def run_backtest(df_tf, tf, cfg, htf_flags=None):
    """
    cfg: dict com toggles (atr/stretch/htf/time_exit/breakeven)
    htf_flags: DF com colunas ['ts','htf_trend_up','htf_slope_up'] para merge_asof (ou None)
    """
    x = add_indicators(df_tf).copy()
    if x.empty or len(x) < (SMA_LONG + 30):
        return pd.DataFrame()

    x = x.sort_values("ts").reset_index(drop=True)

    # Junta flags do HTF (se houver)
    if cfg["use_htf"] and htf_flags is not None and not htf_flags.empty:
        tmp = pd.merge_asof(
            x[["ts"]],
            htf_flags.sort_values("ts"),
            on="ts",
            direction="backward"
        )
        x["htf_trend_up"] = tmp["htf_trend_up"].fillna(False).astype(bool)
        x["htf_slope_up"] = tmp["htf_slope_up"].fillna(False).astype(bool)
    else:
        x["htf_trend_up"] = True
        x["htf_slope_up"] = True

    tick = float(x["close"].iloc[-1] * 0.0001)

    start_idx = max(SMA_LONG + 20, ATR_PERIOD + 5, SLOPE_LOOKBACK + 5)
    trades = []

    for i in range(start_idx, len(x) - 10):
        # Regime base (já existia)
        if not (x["trend_up"].iloc[i] and x["slope_up"].iloc[i]):
            continue

        # HTF alignment
        if cfg["use_htf"]:
            if not x["htf_trend_up"].iloc[i]:
                continue
            if HTF_REQUIRE_SLOPE and (not x["htf_slope_up"].iloc[i]):
                continue

        # Stretch filter (preço esticado)
        if cfg["use_stretch"]:
            dist_l = x["dist_sma_l"].iloc[i]
            if np.isnan(dist_l) or (dist_l > MAX_DIST_SMA_L_PCT):
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

            # Espera preenchimento
            for w in range(1, max_wait + 1):
                curr = x.iloc[i + w]

                if curr["high"] >= entry:
                    filled = True
                    fill_idx = i + w
                    break

                if setup in ["8.2", "8.3"]:
                    # mantém apenas enquanto slope permanecer ok (no LTF)
                    if not x["slope_up"].iloc[i + w]:
                        break

                    # trailing entry/stop
                    if curr["high"] < entry - tick:
                        entry = float(curr["high"] + tick)
                        stop = float(min(stop, curr["low"] - tick))
                else:
                    break

            if not filled:
                continue

            # ATR filter (após ter entry/stop definitivos)
            risk = entry - stop
            atr_val = float(x["atr"].iloc[fill_idx]) if not np.isnan(x["atr"].iloc[fill_idx]) else np.nan
            risk_in_atr = (risk / atr_val) if (atr_val and not np.isnan(atr_val) and atr_val > 0) else np.nan

            if cfg["use_atr"]:
                if np.isnan(risk_in_atr):
                    continue
                if not (ATR_RISK_MIN <= risk_in_atr <= ATR_RISK_MAX):
                    continue

            # Monta linha de trade (meta)
            base_row = {
                "config": cfg["name"],
                "timeframe": tf,
                "setup": setup,

                "signal_ts": x["ts"].iloc[i],
                "fill_ts": x["ts"].iloc[fill_idx],

                "entry": entry,
                "stop": stop,
                "risk": risk,

                "atr": atr_val,
                "risk_in_atr": risk_in_atr,

                "dist_sma_l": float(x["dist_sma_l"].iloc[i]) if not np.isnan(x["dist_sma_l"].iloc[i]) else np.nan,

                "use_atr": cfg["use_atr"],
                "use_stretch": cfg["use_stretch"],
                "use_htf": cfg["use_htf"],
                "use_time_exit": cfg["use_time_exit"],
                "use_breakeven": cfg["use_breakeven"],
            }

            # Resultados por RR (winrate + R real)
            for rr in RRS:
                hit, exit_type, r_res, exit_ts = simulate_trade_per_rr(
                    x=x,
                    fill_idx=fill_idx,
                    entry=entry,
                    initial_stop=stop,
                    rr=rr,
                    cfg=cfg
                )
                base_row[f"hit_{rr}"] = bool(hit)
                base_row[f"exit_type_{rr}"] = exit_type
                base_row[f"R_{rr}"] = r_res
                base_row[f"exit_ts_{rr}"] = exit_ts

            trades.append(base_row)

    return pd.DataFrame(trades)


# =============================
# EXPERIMENT SETUP
# =============================
def build_experiments():
    """
    Comparações A/B:
    - base: tudo OFF
    - atr: só ATR ON
    - stretch: só stretch ON
    - htf: só HTF ON
    - timeexit: só time-exit ON
    - breakeven: só breakeven ON
    - all: tudo ON
    """
    presets = {
        "base":      dict(use_atr=False, use_stretch=False, use_htf=False, use_time_exit=False, use_breakeven=False),
        "atr":       dict(use_atr=True,  use_stretch=False, use_htf=False, use_time_exit=False, use_breakeven=False),
        "stretch":   dict(use_atr=False, use_stretch=True,  use_htf=False, use_time_exit=False, use_breakeven=False),
        "htf":       dict(use_atr=False, use_stretch=False, use_htf=True,  use_time_exit=False, use_breakeven=False),
        "timeexit":  dict(use_atr=False, use_stretch=False, use_htf=False, use_time_exit=True,  use_breakeven=False),
        "breakeven": dict(use_atr=False, use_stretch=False, use_htf=False, use_time_exit=False, use_breakeven=True),
        "all":       dict(use_atr=True,  use_stretch=True,  use_htf=True,  use_time_exit=True,  use_breakeven=True),
    }

    exps = []
    for name in EXPERIMENTS:
        if name not in presets:
            continue
        cfg = {"name": name}
        cfg.update(presets[name])
        exps.append(cfg)
    return exps


def summarize(trades_df):
    if trades_df.empty:
        return pd.DataFrame()

    rows = []
    for (cfg, tf, setup), g in trades_df.groupby(["config", "timeframe", "setup"]):
        row = {"Config": cfg, "TF": tf, "Setup": setup, "Trades": len(g)}
        for rr in RRS:
            hit_rate = g[f"hit_{rr}"].mean() if len(g) else 0.0
            avg_r = g[f"R_{rr}"].mean() if 
            
