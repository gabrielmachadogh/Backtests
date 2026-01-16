import os
import itertools
import numpy as np
import pandas as pd

SYMBOL = os.getenv("SYMBOL", "BTC_USDT")

RRS = [float(x) for x in os.getenv("RRS", "1,1.5,2").split(",")]
PCTS = [int(x) for x in os.getenv("PCTS", "10,15,20,25,30,40,50,60").split(",")]

PAIRWISE_BINS = int(os.getenv("PAIRWISE_BINS", "4"))             # 4 = quartis
PAIRWISE_MAX_FEATURES = int(os.getenv("PAIRWISE_MAX_FEATURES", "25"))  # limita features p/ pairwise (performance)
PAIRWISE_MAX_ROWS = int(os.getenv("PAIRWISE_MAX_ROWS", "50000"))       # 0 = sem limite (cuidado)

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


def melt_outcomes(trades: pd.DataFrame) -> pd.DataFrame:
    rr_cols = [f"rr_{rr}" for rr in RRS]

    base_cols = ["timeframe", "setup", "side"]
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

    # garante as colunas base
    for c in base_cols + ["rr", "outcome"]:
        if c not in long_df.columns:
            raise RuntimeError(f"Coluna obrigatória ausente no long_df: {c}")

    return long_df


def is_discrete_feature(s: pd.Series) -> bool:
    """Heurística para detectar feature discreta (0/1 ou poucos valores)."""
    s2 = pd.to_numeric(s, errors="coerce")
    s2 = s2.dropna()
    if s2.empty:
        return False
    uniq = s2.nunique()
    if uniq <= 6:
        # se for 0/1 ou poucos níveis
        return True
    return False


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


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


def pick_feature_columns(long_df: pd.DataFrame) -> list[str]:
    exclude = set([
        "timeframe", "setup", "side", "rr", "outcome",
        # campos de preço/execução que tendem a não ser “feature de qualidade”
        "entry_price", "stop_price",
    ])

    candidates = []
    for c in long_df.columns:
        if c in exclude:
            continue
        if c.endswith("_time"):
            continue
        # tenta converter
        s = safe_numeric(long_df[c])
        if s.notna().sum() == 0:
            continue
        # remove constantes
        if s.dropna().nunique() <= 1:
            continue
        candidates.append(c)

    # ordena por maior “cobertura” (menos NaN)
    candidates = sorted(candidates, key=lambda c: safe_numeric(long_df[c]).notna().sum(), reverse=True)
    return candidates


def univariate_report(long_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    group_cols = ["timeframe", "setup", "side", "rr"]

    for keys, g in long_df.groupby(group_cols, dropna=False):
        tf, setup, side, rr = keys
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

            # Se for discreta, faz buckets por valor (ex.: 0/1)
            if is_discrete_feature(gg[feat]):
                for v, sub in gg.groupby(gg[feat]):
                    if sub.shape[0] < MIN_TRADES:
                        continue
                    row = {"timeframe": tf, "setup": setup, "side": side, "rr": rr, "feature": feat, "bucket": f"val={v}"}
                    row.update(stats_from_subset(sub))
                    row.update({"thr_lo": v, "thr_hi": v})
                    rows.append(row)
                continue

            # low/high percentis
            for p in PCTS:
                q_lo = gg[feat].quantile(p / 100.0)
                q_hi = gg[feat].quantile(1.0 - p / 100.0)

                low = gg[gg[feat] <= q_lo]
                high = gg[gg[feat] >= q_hi]

                # low
                if low.shape[0] >= MIN_TRADES:
                    row = {"timeframe": tf, "setup": setup, "side": side, "rr": rr, "feature": feat, "bucket": f"low{p}"}
                    row.update(stats_from_subset(low))
                    row.update({"thr_lo": np.nan, "thr_hi": float(q_lo)})
                    rows.append(row)

                # high
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

    # discreto -> usa valor como bin
    if s_ok.nunique() <= 6:
        labels = s.astype("Int64").astype(str)
        return labels, {}

    # contínuo -> qcut + ranges
    try:
        cats, edges = pd.qcut(s, q=n_bins, retbins=True, duplicates="drop")
        # cats é Categorical com intervalos; vamos mapear para 0..k-1
        codes = pd.Series(cats.cat.codes, index=s.index)
        # ranges por bin
        ranges = {}
        for i in range(len(edges) - 1):
            ranges[i] = (float(edges[i]), float(edges[i + 1]))
        return codes, ranges
    except Exception:
        # fallback: usa NaN bins
        return None, {}


def pairwise_report(long_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    group_cols = ["timeframe", "setup", "side", "rr"]

    # limita features para performance
    feats = features[:PAIRWISE_MAX_FEATURES]

    for keys, g in long_df.groupby(group_cols, dropna=False):
        tf, setup, side, rr = keys

        # prepara bins de cada feature
        bin_cache = {}
        range_cache = {}
        for feat in feats:
            b, ranges = make_bins(g[feat], PAIRWISE_BINS)
            if b is None:
                continue
            # precisa ter número de bins razoável
            if b.dropna().nunique() < 2:
                continue
            bin_cache[feat] = b
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

                # ranges (se contínuo). Se discreto, deixa vazio.
                a_lo, a_hi = (np.nan, np.nan)
                b_lo, b_hi = (np.nan, np.nan)

                if isinstance(ba, (int, np.integer)) and ba in range_cache[a]:
                    a_lo, a_hi = range_cache[a][int(ba)]
                if isinstance(bb, (int, np.integer)) and bb in range_cache[b]:
                    b_lo, b_hi = range_cache[b][int(bb)]

                row = {
                    "timeframe": tf,
                    "setup": setup,
                    "side": side,
                    "rr": rr,
                    "feature_a": a,
                    "feature_b": b,
                    "bin_a": str(ba),
                    "bin_b": str(bb),
                    "bin_a_lo": a_lo,
                    "bin_a_hi": a_hi,
                    "bin_b_lo": b_lo,
                    "bin_b_hi": b_hi,
                }
                row.update(st)
                rows.append(row)

    cols = [
        "timeframe", "setup", "side", "rr",
        "feature_a", "feature_b",
        "bin_a", "bin_b",
        "bin_a_lo", "bin_a_hi", "bin_b_lo", "bin_b_hi",
        "trades", "wins", "losses", "win_rate", "win_rate_pct"
    ]
    df = pd.DataFrame(rows, columns=cols)

    if PAIRWISE_MAX_ROWS > 0 and not df.empty:
        df = df.sort_values(["win_rate", "trades"], ascending=[False, False]).head(PAIRWISE_MAX_ROWS)

    return df


def best_patterns_md(uni: pd.DataFrame, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Best patterns (by side) — {SYMBOL}\n\n")
        f.write(f"- MIN_TRADES: {MIN_TRADES}\n")
        f.write(f"- Buckets: low/high {PCTS}\n")
        f.write(f"- Note: thresholds em thr_lo/thr_hi\n\n")

        if uni.empty:
            f.write("Sem dados.\n")
            return

        for (tf, setup, side, rr), g in uni.groupby(["timeframe", "setup", "side", "rr"]):
            g2 = g[g["bucket"] != "ALL"].copy()
            g2 = g2.dropna(subset=["win_rate"])
            g2 = g2.sort_values(["win_rate", "trades"], ascending=[False, False]).head(TOP_K_BEST)

            f.write(f"## {tf} | {setup} | {side} | RR {rr}\n\n")
            f.write("| Feature | Bucket | Trades | WinRate | thr_lo | thr_hi |\n")
            f.write("|---|---|---:|---:|---:|---:|\n")
            for _, r in g2.iterrows():
                thr_lo = "" if pd.isna(r["thr_lo"]) else f"{r['thr_lo']:.6g}"
                thr_hi = "" if pd.isna(r["thr_hi"]) else f"{r['thr_hi']:.6g}"
                f.write(f"| {r['feature']} | {r['bucket']} | {int(r['trades'])} | {r['win_rate_pct']} | {thr_lo} | {thr_hi} |\n")
            f.write("\n")


def main():
    os.makedirs("results", exist_ok=True)

    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError(f"Não achei {INPUT_PATH}. Rode o backtest primeiro.")

    trades = pd.read_csv(INPUT_PATH)

    long_df = melt_outcomes(trades)

    features = pick_feature_columns(long_df)

    # Univariado
    uni = univariate_report(long_df, features)
    uni.to_csv(f"results/patterns_univariate_{SYMBOL}.csv", index=False)

    # Pairwise (com bins + ranges)
    pair = pairwise_report(long_df, features)
    pair.to_csv(f"results/patterns_pairwise_{SYMBOL}.csv", index=False)

    # Best patterns (por side)
    best_patterns_md(uni, f"results/patterns_best_{SYMBOL}.md")


if __name__ == "__main__":
    main()
