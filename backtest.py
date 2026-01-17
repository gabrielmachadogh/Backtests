import os
import time
import requests
import numpy as np
import pandas as pd
from tabulate import tabulate
from itertools import product

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
BE_TARGET_FRACTION = float(os.getenv("BE_TARGET_FRACTION", "0.70"))  # 70% do caminho até o alvo

# ----- Stretch sweep (distâncias em % do preço) -----
# Você pode ajustar essas listas no workflow:
# DIST_SMA8_MAX_LIST="0.02,0.03"
# DIST_SMA80_MAX_LIST="0.05,0.07"
# SMA_GAP_MIN_LIST="0.0,0.01"
DIST_SMA8_MAX_LIST = [float(x) for x in os.getenv("DIST_SMA8_MAX_LIST", "0.02,0.03").split(",")]
DIST_SMA80_MAX_LIST = [float(x) for x in os.getenv("DIST_SMA80_MAX_LIST", "0.05,0.07").split(",")]
SMA_GAP_MIN_LIST = [float(x) for x in os.getenv("SMA_GAP_MIN_LIST", "0.0,0.01").split(",")]

SWEEP_STRETCH = os.getenv("SWEEP_STRETCH", "1") == "1"

# Limite de configs (pra não explodir runtime sem querer)
MAX_CONFIGS = int(os.getenv("MAX_CONFIGS", "200"))

# =============================
# HELPERS
# =============================
def pct_tag(x: float) -> str:
    # 0.02 -> "2p0"
    return f"{x*100:.1f}".replace(".", "p")

def rr_tag(rr: float) -> str:
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


def fetch_history(symbol):
    print(f"Baixando histórico 1h para {symbol}...")
    all_dfs = []
    end_ts = int(time.time())
    step = WINDOW_DAYS * 86400

    total = 0
    for n in range(80):
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

    mapping = {"2h": "2H", "4h": "4H", "1d": "1D", "1w": "1W"}  # W-SUN
    if rule not in mapping:
        return df

    dfi = df.set_index("ts")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    res = dfi.resample(mapping[rule]).agg(agg).dropna()
    return res.reset_index()


# =============================
# INDICATORS
# =============================
def add_indicators(df):
    x = df.copy()
    x["sma_s"] = x["close"].rolling(SMA_SHORT).mean()
    x["sma_l"] = x["close"].rolling(SMA_LONG).mean()

    # Regime (long only)
    x["trend_up"] = (x["close"] > x["sma_s"]) & (x["close"] > x["sma_l"]) & (x["sma_s"] > x["sma_l"])
    x["slope_up"] = x["sma_s"] > x["sma_s"].shift(1).rolling(SLOPE_LOOKBACK).max()

    # Distâncias (% do preço)
    x["dist_sma8"] = (x["close"] - x["sma_s"]) / x["close"]
    x["dist_sma80"] = (x["close"] - x["sma_l"]) / x["close"]
    x["sma_gap"] = (x["sma_s"] - x["sma_l"]) / x["close"]

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
    Retorna:
      hit_target (bool), exit_type (str), r_result (float), exit_ts (Timestamp)
    """
    risk = entry - initial_stop
    if risk <= 0 or np.isnan(risk):
        return False, "invalid", np.nan, pd.NaT

    stop = initial_stop
    target = entry + (risk * rr)

    # Break-even: 70% do caminho até o alvo (depende do RR)
    be_trigger = entry + (target - entry) * BE_TARGET_FRACTION  # entry + risk*rr*0.7
    be_pending = False
    be_active = False

    n_bars = max(int(TIME_EXIT_BARS), 1)
    last_idx = min(fill_idx + n_bars - 1, len(x) - 1)

    for k in range(fill_idx, last_idx + 1):
        c = x.iloc[k]

        # Aplica BE somente a partir do candle seguinte ao trigger (evita ambiguidade intrabar)
        if cfg["use_breakeven"] and be_pending and not be_active:
            stop = entry
            be_active = True
            be_pending = False

        # Ordem conservadora intrabar:
        # 1) stop
        # 2) target
        # 3) trigger do BE (para ativar no próximo candle)
        if c["low"] <= stop:
            r = (stop - entry) / risk  # -1.0 ou 0.0 (se stop virou entry)
            return False, "stop" if stop < entry else "breakeven_stop", float(r), c["ts"]

        if c["high"] >= target:
            return True, "target", float(rr), c["ts"]

        if cfg["use_breakeven"] and (not be_active) and (c["high"] >= be_trigger):
            be_pending = True

    # Não bateu stop/target dentro de N candles
    if cfg["use_time_exit"]:
        exit_px = float(x["close"].iloc[last_idx])
        r = (exit_px - entry) / risk
        return False, "time_exit", float(r), x["ts"].iloc[last_idx]

    # comportamento antigo: vira loss (comparável ao seu base)
    return False, "timeout_loss", -1.0, x["ts"].iloc[last_idx]


# =============================
# BACKTEST CORE
# =============================
def run_backtest(df_tf, tf, cfg, htf_flags=None):
    x = add_indicators(df_tf).copy()
    if x.empty or len(x) < (SMA_LONG + 30):
        return pd.DataFrame()

    x = x.sort_values("ts").reset_index(drop=True)

    # HTF merge
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
    start_idx = max(SMA_LONG + 20, SLOPE_LOOKBACK + 5)

    trades = []

    for i in range(start_idx, len(x) - 10):
        # Regime base
        if not (x["trend_up"].iloc[i] and x["slope_up"].iloc[i]):
            continue

        # HTF alignment
        if cfg["use_htf"]:
            if not x["htf_trend_up"].iloc[i]:
                continue
            if HTF_REQUIRE_SLOPE and (not x["htf_slope_up"].iloc[i]):
                continue

        # Stretch filters (sweep)
        if cfg["use_stretch"]:
            dist8 = x["dist_sma8"].iloc[i]
            dist80 = x["dist_sma80"].iloc[i]
            gap = x["sma_gap"].iloc[i]

            if np.isnan(dist8) or np.isnan(dist80) or np.isnan(gap):
                continue

            # Como trend_up exige close > sma, dist tende a ser positivo;
            # regra é: "não operar se estiver esticado acima de X%"
            if dist8 > cfg["dist_sma8_max"]:
                continue
            if dist80 > cfg["dist_sma80_max"]:
                continue
            if gap < cfg["sma_gap_min"]:
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

            base_row = {
                "config": cfg["name"],
                "timeframe": tf,
                "setup": setup,

                "signal_ts": x["ts"].iloc[i],
                "fill_ts": x["ts"].iloc[fill_idx],

                "entry": entry,
                "stop": stop,
                "risk": risk,

                "dist_sma8": float(x["dist_sma8"].iloc[i]),
                "dist_sma80": float(x["dist_sma80"].iloc[i]),
                "sma_gap": float(x["sma_gap"].iloc[i]),

                "use_htf": cfg["use_htf"],
                "use_time_exit": cfg["use_time_exit"],
                "use_breakeven": cfg["use_breakeven"],
                "use_stretch": cfg["use_stretch"],

                "dist_sma8_max": cfg.get("dist_sma8_max", np.nan),
                "dist_sma80_max": cfg.get("dist_sma80_max", np.nan),
                "sma_gap_min": cfg.get("sma_gap_min", np.nan),
            }

            for rr in RRS:
                hit, exit_type, r_res, exit_ts = simulate_trade_per_rr(
                    x=x,
                    fill_idx=fill_idx,
                    entry=entry,
                    initial_stop=stop,
                    rr=rr,
                    cfg=cfg
                )
                base_row[f"hit_{rr_tag(rr)}"] = bool(hit)
                base_row[f"exit_type_{rr_tag(rr)}"] = exit_type
                base_row[f"R_{rr_tag(rr)}"] = r_res
                base_row[f"exit_ts_{rr_tag(rr)}"] = exit_ts

            trades.append(base_row)

    return pd.DataFrame(trades)


# =============================
# EXPERIMENTS (A/B + sweep)
# =============================
def build_experiments():
    """
    Variáveis ON/OFF:
      - HTF
      - TIME EXIT
      - BREAK EVEN
    Stretch:
      - OFF (baseline)
      - ON com sweep de (dist_sma8_max, dist_sma80_max, sma_gap_min)
    """
    scenarios = []
    for use_htf, use_time_exit, use_breakeven in product([False, True], repeat=3):
        base_name = f"htf{int(use_htf)}_tx{int(use_time_exit)}_be{int(use_breakeven)}"
        scenarios.append({
            "name": base_name,
            "use_htf": use_htf,
            "use_time_exit": use_time_exit,
            "use_breakeven": use_breakeven,
            "use_stretch": False,
        })

    configs = []

    # Sempre inclui stretch OFF
    configs.extend(scenarios)

    # Stretch ON (sweep)
    if SWEEP_STRETCH:
        for sc in scenarios:
            for s8, s80, gap in product(DIST_SMA8_MAX_LIST, DIST_SMA80_MAX_LIST, SMA_GAP_MIN_LIST):
                name = (
                    f"{sc['name']}_st1"
                    f"_s8{pct_tag(s8)}"
                    f"_s80{pct_tag(s80)}"
                    f"_gap{pct_tag(gap)}"
                )
                cfg = dict(sc)
                cfg["name"] = name
                cfg["use_stretch"] = True
                cfg["dist_sma8_max"] = float(s8)
                cfg["dist_sma80_max"] = float(s80)
                cfg["sma_gap_min"] = float(gap)
                configs.append(cfg)

    # Cap de segurança
    if len(configs) > MAX_CONFIGS:
        print(f"Aviso: configs={len(configs)} > MAX_CONFIGS={MAX_CONFIGS}. Cortando lista.")
        configs = configs[:MAX_CONFIGS]

    return configs


def summarize(trades_df):
    if trades_df.empty:
        return pd.DataFrame()

    rows = []
    for (cfg, tf, setup), g in trades_df.groupby(["config", "timeframe", "setup"]):
        row = {"Config": cfg, "TF": tf, "Setup": setup, "Trades": len(g)}
        for rr in RRS:
            k = rr_tag(rr)
            hit_rate = g[f"hit_{k}"].mean() if len(g) else 0.0
            avg_r = g[f"R_{k}"].mean() if len(g) else np.nan
            row[f"WR {k}"] = f"{hit_rate:.1%}"
            row[f"AvgR {k}"] = f"{avg_r:.2f}"
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["Config", "TF", "Setup"])


# =============================
# MAIN
# =============================
def main():
    os.makedirs("results", exist_ok=True)

    df_raw = fetch_history(SYMBOL)
    if df_raw.empty:
        print("Erro: Sem dados baixados.")
        pd.DataFrame({"status": ["no_data"]}).to_csv(f"results/sweep_trades_{SYMBOL}.csv", index=False)
        return

    # OHLCV por timeframe
    tf_dfs = {"1h": df_raw.copy()}
    for tf in TIMEFRAMES:
        if tf == "1h":
            continue
        tf_dfs[tf] = resample(df_raw, tf)

    if SAVE_OHLCV:
        for tf, d in tf_dfs.items():
            if d is None or d.empty:
                continue
            d.to_csv(f"results/ohlcv_{SYMBOL}_{tf}.csv", index=False)

    # Indicadores nos HTFs (para merge_asof)
    tf_ind = {}
    for tf, d in tf_dfs.items():
        if d is None or d.empty:
            tf_ind[tf] = pd.DataFrame()
        else:
            tf_ind[tf] = add_indicators(d).sort_values("ts").reset_index(drop=True)

    configs = build_experiments()
    print(f"Total de configs: {len(configs)} (SWEEP_STRETCH={SWEEP_STRETCH})")

    all_trades = []
    for cfg in configs:
        if DEBUG:
            print(f"\n=== Config: {cfg['name']} ===")

        for tf in TIMEFRAMES:
            df_tf = tf_dfs.get(tf, pd.DataFrame())
            if df_tf is None or df_tf.empty or len(df_tf) < 120:
                continue

            # HTF flags
            htf_tf = HTF_MAP.get(tf)
            htf_flags = None
            if cfg["use_htf"] and htf_tf:
                h = tf_ind.get(htf_tf, pd.DataFrame())
                if h is not None and not h.empty:
                    htf_flags = h[["ts", "trend_up", "slope_up"]].copy()
                    htf_flags = htf_flags.rename(columns={
                        "trend_up": "htf_trend_up",
                        "slope_up": "htf_slope_up",
                    })

            try:
                t = run_backtest(df_tf=df_tf, tf=tf, cfg=cfg, htf_flags=htf_flags)
                if not t.empty:
                    all_trades.append(t)
            except Exception as e:
                print(f"Erro em {cfg['name']} / {tf}: {e}")

    if not all_trades:
        print("0 trades gerados.")
        pd.DataFrame({"status": ["no_trades"]}).to_csv(f"results/sweep_trades_{SYMBOL}.csv", index=False)
        return

    final = pd.concat(all_trades, ignore_index=True)

    # CSV completo
    final.to_csv(f"results/sweep_trades_{SYMBOL}.csv", index=False)

    # CSV por config (facilita comparar)
    for cfg_name, g in final.groupby("config"):
        g.to_csv(f"results/sweep_trades_{SYMBOL}_{cfg_name}.csv", index=False)

    # Resumo
    sum_df = summarize(final)
    print("\n" + tabulate(sum_df, headers="keys", tablefmt="grid", showindex=False))

    with open(f"results/sweep_summary_{SYMBOL}.md", "w", encoding="utf-8") as f:
        f.write(f"# Sweep Summary - {SYMBOL}\n\n")
        f.write("Params:\n\n")
        f.write(f"- TIME_EXIT_BARS={TIME_EXIT_BARS}\n")
        f.write(f"- BE_TARGET_FRACTION={BE_TARGET_FRACTION}\n")
        f.write(f"- SWEEP_STRETCH={SWEEP_STRETCH}\n")
        f.write(f"- DIST_SMA8_MAX_LIST={DIST_SMA8_MAX_LIST}\n")
        f.write(f"- DIST_SMA80_MAX_LIST={DIST_SMA80_MAX_LIST}\n")
        f.write(f"- SMA_GAP_MIN_LIST={SMA_GAP_MIN_LIST}\n")
        f.write(f"- HTF_MAP={HTF_MAP}, require_slope={HTF_REQUIRE_SLOPE}\n\n")
        f.write(tabulate(sum_df, headers="keys", tablefmt="pipe", showindex=False))

    sum_df.to_csv(f"results/sweep_summary_{SYMBOL}.csv", index=False)


if __name__ == "__main__":
    main()
