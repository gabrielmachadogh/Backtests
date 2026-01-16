import os
import itertools
import numpy as np
import pandas as pd

SYMBOL = os.getenv("SYMBOL", "BTC_USDT")

RRS = [float(x) for x in os.getenv("RRS", "1,1.5,2").split(",")]
PCTS = [int(x) for x in os.getenv("PCTS", "10,15,20,25,30,40,50,60").split(",")]

PAIRWISE_BINS = int(os.getenv("PAIRWISE_BINS", "4"))                  # 4 = quartis
PAIRWISE_MAX_FEATURES = int(os.getenv("PAIRWISE_MAX_FEATURES", "25")) # limita features p/ pairwise
PAIRWISE_MAX_ROWS = int(os.getenv("PAIRWISE_MAX_ROWS", "50000"))      # 0 = sem limite

MIN_TRADES = int(os.getenv("MIN_TRADES", "30"))
TOP_K_BEST = int(os.getenv("TOP_K_BEST", "20"))

INPUT_PATH = f"results/backtest_trades_{SYMBOL}.csv"


def fmt_pct(win_rate) -> str:
    try:
        if win_rate is None or (isinstance(win_rate, float) and np.isnan(win_rate)):
            return "-"
        pct = float(win_rate) * 100.0
        if pct >= 100:
            return "100%"
        s = f"{pct:.1f}".replace(".", ",")
        if s.endswith(",0"):
            s = s[:-2]
        return f"{s}%"
    except Exception:
        return "-"


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def melt_outcomes(trades: pd.DataFrame) -> pd.DataFrame:
    rr_cols = [f"rr_{rr}" for rr in RRS]

    keep_cols = [c for c in trades.columns if c not in ["signal_time", "entry_time"]]
    df = trades[keep_cols].copy()

    long_df = df.melt(
        id_vars=[c for c in keep_cols if c not in rr_cols],
        value_vars=rr_cols,
        var_name="rr_col",
        value_name="outcome",
    )
    long_df["rr"] = long_df["rr_col"].str.replace("rr_", "", regex=False).astype(float)
    long_df = long_df.drop(columns=["rr_col"])

    # só resolvidos para winrate
    long_df = long_df[long_df["outcome"].isin(["win", "loss"])].copy()

    for c in ["timeframe", "setup", "side", "rr", "outcome"]:
        if c not in long_df.columns:
            raise RuntimeError(f"Coluna obrigatória ausente: {c}")

    return long_df


def pick_feature_columns(long_df: pd.DataFrame) -> list[str]:
    exclude = set([
        "timeframe", "setup", "side", "rr", "outcome",
        "entry_price", "stop_price",
    ])

    candidates = []
    for c in long_df.columns:
        if c in exclude:
            continue
        if c.endswith("_time"):
            continue

        s = safe_numeric(long_df[c])
        if s.notna().sum() == 0:
            continue
        if s.dropna().nunique() <= 1:
            continue

        candidates.append(c)

    candidates = sorted(candidates, key=lambda c: safe_numeric(long_df[c]).notna().sum(), reverse=True)
    return candidates


def is_discrete_feature(series: pd.Series) -> bool:
    s = safe_numeric(series).dropna()
    if s.empty:
        return False
    # poucos valores -> tratar como discreto (0/1, flags, categorias numéricas)
    return s.nunique() <= 6


def stats_from_subset(sub: pd.DataFrame) -> dict:
    trades_n = int(sub.shape[0])
    wins = int((sub["outcome"] == "win").sum())
    losses = int((sub["outcome"] == "loss").sum())
    denom = wins + losses
    win_rate = (wins / denom) if denom > 0 else np.nan
    return {
        "trades": trades_n,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "win_rate_pct": fmt_pct(win_rate),
    }


def univariate_report(long_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    group_cols = ["timeframe", "setup", "side", "rr"]

    for (tf, setup, side, rr), g in long_df.groupby(group_cols, dropna=False):
        for feat in features:
            s = safe_numeric(g[feat])
            gg = g.copy()
            gg[feat] = s
            gg = gg.dropna(subset=[feat])
            if gg.shape[0] < MIN_TRADES:
                continue

            # ALL
            base = {"timeframe": tf, "setup": setup, "side": side, "rr": rr, "feature": feat, "bucket": "ALL"}
            base.update(stats_from_subset(gg))
            base.update({"thr_lo": np.nan, "thr_hi": np.nan})
            rows.append(base)

            # discreto -> buckets por valor (sem converter para int!)
            if is_discrete_feature(gg[feat]):
                # usa string segura do valor
                values = gg[feat].map(lambda x: f"{float(x):g}" if pd.notna(x) else np.nan)
                gg2 = gg.copy()
                gg2["_val"] = values

                for v, sub in gg2.groupby("_val", dropna=True):
                    if sub.shape[0] < MIN_TRADES:
                        continue
                    row = {"timeframe": tf, "setup": setup, "side": side, "rr": rr, "feature": feat, "bucket": f"val={v}"}
                    row.update(stats_from_subset(sub))
                    # thr_lo/thr_hi só faz sentido em contínuas; aqui guarda o valor
                    try:
                        vv = float(v)
                    except Exception:
                        vv = np.nan
                    row.update({"thr_lo": vv, "thr_hi": vv})
                    rows.append(row)

                continue

            # low/high percentis (contínuas)
            for p in PCTS:
                q_lo = gg[feat].quantile(p / 100.0)
                q_hi = gg[feat].quantile(1.0 - p / 100.0)

                low = gg[gg[feat] <= q_lo]
                high = gg[gg[feat] >= q_hi]

                if low.shape[0] >= MIN_TRADES:
                    row = {"timeframe": tf, "setup": setup, "side": side, "rr": rr, "feature": feat, "bucket": f"low{p}"}
                    row.update(stats_from_subset(low))
                    row.update({"thr_lo": np.nan, "thr_hi": float(q_lo)})
                    rows.append(row)

                if high.shape[0] >= MIN_TRADES:
                    row = {"timeframe": tf, "setup": setup, "side": side, "rr": rr, "feature": feat, "bucket": f"high{p}"}
                    row.update(stats_from_subset(high))
                    row.update({"thr_lo": float(q_hi), "thr_hi": np.nan})
                    rows.append(row)

    cols = ["timeframe", "setup", "side", "rr", "feature", "bucket", "trades", "wins", "losses", "win_rate", "win_rate_pct", "thr_lo", "thr_hi"]
    return pd.DataFrame(rows, columns=cols)


def make_bins(series: pd.Series, n_bins: int):
    """
    Retorna (bin_labels, bin_ranges)
      - bin_labels: Series com labels (int bins ou string valores)
      - bin_ranges: dict bin_label -> (lo, hi) para contínuas; vazio para discretas
    """
    s = safe_numeric(series)

    s_ok = s.dropna()
    if s_ok.empty:
        return None, {}

    # discreto -> usar string do valor (SEM astype Int64)
    if s_ok.nunique() <= 6:
        labels = s.map(lambda x: f"{float(x):g}" if pd.notna(x) else np.nan)
        return labels, {}

    # contínuo -> qcut + ranges
    try:
        cats, edges = pd.qcut(s, q=n_bins, retbins=True, duplicates="drop")
        codes = pd.Series(cats.cat.codes, index=s.index)
        ranges = {}
        for i in range(len(edges) - 1):
            ranges[i] = (float(edges[i]), float(edges[i + 1]))
        return codes, ranges
    except Exception:
        return None, {}


def pairwise_report(long_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    group_cols = ["timeframe", "setup", "side", "rr"]

    feats = features[:PAIRWISE_MAX_FEATURES]

    for (tf, setup, side, rr), g in long_df.groupby(group_cols, dropna=False):
        bin_cache = {}
        range_cache = {}

        for feat in feats:
            bins, ranges = make_bins(g[feat], PAIRWISE_BINS)
            if bins is None:
                continue
            if bins.dropna().nunique() < 2:
                continue
            bin_cache[feat] = bins
            range_cache[feat] = ranges

        feats_ok = list(bin_cache.keys())
        if len(feats_ok) < 2:
            continue

        for a, b in itertools.combinations(feats_ok, 2):
            tmp = g.copy()
            tmp["bin_a"] = bin_cache[a]
            tmp["bin_b"] = bin_cache[b]
            tmp = tmp.dropna(subset=["bin_a", "bin_b"])

            if tmp.shape[0] < MIN_TRADES:
                continue

            for (ba, bb), sub in tmp.groupby(["bin_a", "bin_b"]):
                if sub.shape[0] < MIN_TRADES:
                    continue

                st = stats_from_subset(sub)

                a_lo, a_hi = (np.nan, np.nan)
                b_lo, b_hi = (np.nan, np.nan)

                # ranges só existem para contínuas (bins int)
                if isinstance(ba, (int, np.integer)) and int(ba) in range_cache[a]:
                    a_lo, a_hi = range_cache[a][int(ba)]
                if isinstance(bb, (int, 
