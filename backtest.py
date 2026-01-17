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
RANK_RR = float(os.getenv("RANK_RR", "1.5"))  # métrica principal para rankear configs/strech

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
TIME_EXIT_BARS = int(os.getenv("TIME_EXIT_BARS", "50"))  # N candles (por TF)
BE_TARGET_FRACTION = float(os.getenv("BE_TARGET_FRACTION", "0.70"))  # 70% do alvo

# ----- Stretch sweep (distâncias em % do SMA, mais interpretável) -----
# dist_sma8  = (close - sma8) / sma8
# dist_sma80 = (close - sma80) / sma80
# sma_gap    = (sma8 - sma80) / sma80
DIST_SMA8_MAX_LIST = [float(x) for x in os.getenv("DIST_SMA8_MAX_LIST", "0.01,0.02,0.03").split(",")]
DIST_SMA80_MAX_LIST = [float(x) for x in os.getenv("DIST_SMA80_MAX_LIST", "0.03,0.05,0.07").split(",")]
SMA_GAP_MIN_LIST = [float(x) for x in os.getenv("SMA_GAP_MIN_LIST", "0.00,0.005,0.01").split(",")]

# Estágio 2: quantos configs do estágio 1 vão para o sweep
TOP_K_CONFIGS_FOR_SWEEP = int(os.getenv("TOP_K_CONFIGS_FOR_SWEEP", "2"))

# Split treino/teste (tempo)
TRAIN_FRACTION = float(os.getenv("TRAIN_FRACTION", "0.70"))

# Para evitar conclusões com amostra pequena
MIN_TRADES_FOR_RANK = int(os.getenv("MIN_TRADES_FOR_RANK", "80"))

# =============================
# HELPERS
# =============================
def rr_key(rr: float) -> str:
    # Mantém como string simples para colunas
    if float(rr).is_integer():
        return str(int(rr))
    return str(rr)

def pct_tag(x: float) -> str:
    # 0.02 -> "2p0"
    return f"{x*100:.1f}".replace(".", "p")

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

    mapping = {"2h": "2H", "4h": "4H", "1d": "1D", "1w": "1W"}  # W-SUN por padrão
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
    x["sma8"] = x["close"].rolling(SMA_SHORT).mean()
    x["sma80"] = x["close"].rolling(SMA_LONG).mean()

    # Regime (long only)
    x["trend_up"] = (x["close"] > x["sma8"]) & (x["close"] > x["sma80"]) & (x["sma8"] > x["sma80"])
    x["slope_up"] = x["sma8"] > x["sma8"].shift(1).rolling(SLOPE_LOOKBACK).max()

    # Stretch features (em % do SMA -> mais estável)
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

    be_trigger = entry + (target - entry) * BE_TARGET_FRACTION  # 70% do caminho até o alvo
    be_pending = False
    be_active = False

    n_bars = max(int(TIME_EXIT_BARS), 1)
    last_idx = min(fill_idx + n_bars - 1, len(x) - 1)

    for k in range(fill_idx, last_idx + 1):
        c = x.iloc[k]

        # aplica BE no início do candle (se foi acionado no candle anterior)
        if cfg["use_breakeven"] and be_pending and not be_active:
            stop = entry
            be_active = True
            be_pending = False

        # ordem conservadora intrabar:
        # 1) stop
        # 2) target
        # 3) acionamento do BE (para vigorar só no próximo candle)
        if c["low"] <= stop:
            r = (stop - entry) / risk  # -1 ou 0
            return False, "stop" if stop < entry else "breakeven_stop", float(r), c["ts"]

        if c["high"] >= target:
            return True, "target", float(rr), c["ts"]

        if cfg["use_breakeven"] and (not be_active) and (c["high"] >= be_trigger):
            be_pending = True

    # não bateu stop/target dentro do horizonte
    if cfg["use_time_exit"]:
        exit_px = float(x["close"].iloc[last_idx])
        r = (exit_px - entry) / risk
        return False, "time_exit", float(r), x["ts"].iloc[last_idx]

    # modo antigo: timeout vira loss
    return False, "timeout_loss", -1.0, x["ts"].iloc[last_idx]

# =============================
# BACKTEST CORE (gera trades "brutos" - sem stretch filter)
# =============================
def run_backtest_generate_trades(df_tf, tf, cfg, htf_flags=None):
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
        # regime base
        if not (x["trend_up"].iloc[i] and x["slope_up"].iloc[i]):
            continue

        # HTF
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

            # features para stretch (no candle do sinal)
            dist8 = x["dist_sma8"].iloc[i]
            dist80 = x["dist_sma80"].iloc[i]
            gap = x["sma_gap"].iloc[i]
            if np.isnan(dist8) or np.isnan(dist80) or np.isnan(gap):
                continue

            row = {
                "config": cfg["name"],
                "timeframe": tf,
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

            # resultados por RR (já simulados)
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
def build_structural_configs():
    """
    Só o que você pediu manter como variável:
      - HTF on/off
      - time-exit on/off
      - break-even on/off (70% do alvo)
    Sem stretch aqui (stretch vai ser aplicado depois por filtro).
    """
    configs = []
    for use_htf, use_time_exit, use_breakeven in product([False, True], repeat=3):
        name = f"htf{int(use_htf)}_tx{int(use_time_exit)}_be{int(use_breakeven)}"
        configs.append({
            "name": name,
            "use_htf": use_htf,
            "use_time_exit": use_time_exit,
            "use_breakeven": use_breakeven,
        })
    return configs

# =============================
# STRETCH FILTER (aplica depois, sem rerodar backtest)
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

    return pd.DataFrame(rows).sort_values(by_cols + ["Trades"], ascending=[True]*len(by_cols) + [False])

# =============================
# MAIN
# =============================
def main():
    os.makedirs("results", exist_ok=True)

    df_raw = fetch_history(SYMBOL)
    if df_raw.empty:
        print("Erro: Sem dados baixados.")
        pd.DataFrame({"status": ["no_data"]}).to_csv(f"results/conclusive_{SYMBOL}.csv", index=False)
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

    # indicadores por tf (para HTF flags)
    tf_ind = {}
    for tf, d in tf_dfs.items():
        if d is None or d.empty:
            tf_ind[tf] = pd.DataFrame()
        else:
            tf_ind[tf] = add_indicators(d).sort_values("ts").reset_index(drop=True)

    configs = build_structural_configs()
    print(f"Configs estruturais (HTF/TX/BE): {len(configs)}")

    # 1) gera trades brutos por config (SEM stretch)
    all_trades = []
    for cfg in configs:
        print(f"Gerando trades: {cfg['name']} ...")
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
                    htf_flags = h[["ts", "trend_up", "slope_up"]].copy().rename(columns={
                        "trend_up": "htf_trend_up",
                        "slope_up": "htf_slope_up",
                    })

            try:
                t = run_backtest_generate_trades(df_tf=df_tf, tf=tf, cfg=cfg, htf_flags=htf_flags)
                if not t.empty:
                    all_trades.append(t)
            except Exception as e:
                print(f"Erro em {cfg['name']} / {tf}: {e}")

    if not all_trades:
        print("0 trades gerados.")
        pd.DataFrame({"status": ["no_trades"]}).to_csv(f"results/conclusive_{SYMBOL}.csv", index=False)
        return

    trades = pd.concat(all_trades, ignore_index=True)
    trades.to_csv(f"results/conclusive_trades_raw_{SYMBOL}.csv", index=False)

    # Split global (por tempo)
    train, test, cutoff = split_train_test(trades)
    print(f"Cutoff treino/teste: {cutoff}")

    # 2) ESTÁGIO 1: escolher melhor config (sem stretch)
    stage1_rows = []
    for cfg_name, g in trades.groupby("config"):
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
        })

    stage1 = pd.DataFrame(stage1_rows)
    stage1["Eligible"] = stage1["Test Trades"] >= MIN_TRADES_FOR_RANK
    stage1 = stage1.sort_values(["Eligible", "Test AvgR", "Test Trades"], ascending=[False, False, False])

    stage1.to_csv(f"results/conclusive_stage1_structural_{SYMBOL}.csv", index=False)
    print("\n=== Stage 1 (sem stretch) - ranking por Test AvgR @ RR="
          f"{RANK_RR} (min trades={MIN_TRADES_FOR_RANK}) ===")
    print(tabulate(stage1.head(20), headers="keys", tablefmt="grid", showindex=False))

    # Seleciona top K configs elegíveis (ou as melhores mesmo se nenhuma elegível)
    eligible = stage1[stage1["Eligible"]].copy()
    if eligible.empty:
        top_cfgs = stage1.head(TOP_K_CONFIGS_FOR_SWEEP)["Config"].tolist()
    else:
        top_cfgs = eligible.head(TOP_K_CONFIGS_FOR_SWEEP)["Config"].tolist()

    print(f"\nConfigs selecionados para sweep de stretch (Stage 2): {top_cfgs}")

    # 3) ESTÁGIO 2: sweep de stretch nos melhores configs (treino escolhe, teste valida)
    stretch_grid = list(product(DIST_SMA8_MAX_LIST, DIST_SMA80_MAX_LIST, SMA_GAP_MIN_LIST))
    stage2_rows = []
    best_rows = []

    for cfg_name in top_cfgs:
        base_train = train[train["config"] == cfg_name].copy()
        base_test = test[test["config"] == cfg_name].copy()

        base_train_m = metrics_overall(base_train, RANK_RR)
        base_test_m = metrics_overall(base_test, RANK_RR)

        best = None  # (train_avgR, params...)
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

            # mínimo de trades no treino também (para evitar escolher coisa com 5 trades)
            if m["Trades"] < MIN_TRADES_FOR_RANK:
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

        # Se não achou nenhum com trades suficientes, pega o melhor independente
        if best is None:
            tmp = pd.DataFrame([r for r in stage2_rows if r["Config"] == cfg_name]).copy()
            tmp = tmp.sort_values(["Train AvgR", "Train Trades"], ascending=[False, False]).head(1)
            best = tmp.iloc[0].to_dict()

        # Avalia no TESTE com o melhor stretch
        best_test_filtered = apply_stretch_filter(base_test, best["dist_sma8_max"], best["dist_sma80_max"], best["sma_gap_min"])
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

        # Também salva um resumo por TF+Setup no TESTE para o melhor stretch (mais diagnóstico)
        per = summary_by(best_test_filtered, RANK_RR, ["timeframe", "setup"])
        per.to_csv(f"results/conclusive_stage2_best_per_tf_setup_{SYMBOL}_{cfg_name}.csv", index=False)

    stage2 = pd.DataFrame(stage2_rows)
    stage2.to_csv(f"results/conclusive_stage2_stretch_sweep_{SYMBOL}.csv", index=False)

    best_df = pd.DataFrame(best_rows).sort_values("BEST Test AvgR", ascending=False)
    best_df.to_csv(f"results/conclusive_stage2_best_stretch_{SYMBOL}.csv", index=False)

    print("\n=== Stage 2 (stretch) - melhores parâmetros escolhidos no TREINO e validados no TESTE ===")
    print(tabulate(best_df, headers="keys", tablefmt="grid", showindex=False))

    # Markdown resumo
    with open(f"results/conclusive_summary_{SYMBOL}.md", "w", encoding="utf-8") as f:
        f.write(f"# Conclusive Summary - {SYMBOL}\n\n")
        f.write(f"- Train fraction: {TRAIN_FRACTION}\n")
        f.write(f"- Rank RR: {RANK_RR}\n")
        f.write(f"- Min trades for rank: {MIN_TRADES_FOR_RANK}\n")
        f.write(f"- Time exit bars: {TIME_EXIT_BARS}\n")
        f.write(f"- BE target fraction: {BE_TARGET_FRACTION}\n")
        f.write(f"- HTF require slope: {HTF_REQUIRE_SLOPE}\n")
        f.write(f"- Stretch grid:\n")
        f.write(f"  - DIST_SMA8_MAX_LIST={DIST_SMA8_MAX_LIST}\n")
        f.write(f"  - DIST_SMA80_MAX_LIST={DIST_SMA80_MAX_LIST}\n")
        f.write(f"  - SMA_GAP_MIN_LIST={SMA_GAP_MIN_LIST}\n\n")
        f.write("## Stage 1 (Structural, no stretch)\n\n")
        f.write(tabulate(stage1.head(20), headers="keys", tablefmt="pipe", showindex=False))
        f.write("\n\n## Stage 2 (Best stretch per selected config)\n\n")
        f.write(tabulate(best_df, headers="keys", tablefmt="pipe", showindex=False))

if __name__ == "__main__":
    main()
