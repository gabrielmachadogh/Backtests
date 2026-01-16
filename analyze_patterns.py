import os
import itertools
import numpy as np
import pandas as pd

SYMBOL = os.getenv("SYMBOL", "BTC_USDT")

RRS = [float(x) for x in os.getenv("RRS", "1,1.5,2").split(",")]
PCTS = [int(x) for x in os.getenv("PCTS", "10,15,20,25,30,40,50,60").split(",")]
PAIRWISE_BINS = int(os.getenv("PAIRWISE_BINS", "4"))  # 4 = quartis
MIN_TRADES = int(os.getenv("MIN_TRADES", "30"))       # evita padrões com amostra ridícula
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
    base_cols = ["timeframe", "setup", "side"]  # básicos
    # mantém todas as features numéricas
    feature_cols = [c for c in trades.columns if c not in base_cols + rr_cols and c not in ["signal_time", "entry_time"]]
    df = trades[base_cols + feature_cols + rr_cols].copy()
    long_df = df.melt(
        id_vars=base_cols + feature_cols,
        value_vars=rr_cols,
        var_name="rr_col",
        value_name="outcome",
    )
    long_df["rr"] = long_df["rr_col"].str.replace("rr_", "", regex=False).astype(float)
    long_df = long_df.drop(columns=["rr_col"])
    # só resolvidos para winrate
    long_df = long_df[long_df["outcome"].isin(["win", "loss"])].copy()
    return long_df


def univariate_report(long_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for (tf, setup, rr), g in long_df.groupby(["timeframe", "setup", "rr"], dropna=False):
        for feat in features:
            if feat not in g.columns:
                continue
            s = pd.to_numeric(g[feat], errors="coerce")
            gg = g.copy()
            gg[feat] = s
            gg = gg.dropna(subset=[feat])
            if gg.shape[0] < MIN_TRADES:
                continue

            # ALL
            wins = int((gg["outcome"] == "win").sum())
            losses = int((gg["outcome"] == "loss").sum())
            denom = wins + losses
            wr = wins / denom if denom > 0 else np.nan
            rows.append({
                "timeframe": tf, "setup": setup, "rr": rr,
                "feature": feat, "bucket": "ALL",
                "trades": int(gg.shape[0]), "wins": wins, "losses": losses,
                "win_rate": wr, "win_rate_pct": fmt_pct(wr),
            })

            for p in PCTS:
                q_lo = gg[feat].quantile(p / 100.0)
                q_hi = gg[feat].quantile(1.0 - p / 100.0)

                low = gg[gg[feat] <= q_lo]
                high = gg[gg[feat] >= q_hi]

                for name, sub in [(f"low{p}", low), (f"high{p}", high)]:
                    if sub.shape[0] < MIN_TRADES:
                        continue
                    w = int((sub["outcome"] == "win").sum())
                    l = int((sub["outcome"] == "loss").sum())
                    d = w + l
                    wr2 = w / d if d > 0 else np.nan
                    rows.append({
                        "timeframe": tf, "setup": setup, "rr": rr,
                        "feature": feat, "bucket": name,
                        "trades": int(sub.shape[0]), "wins": w, "losses": l,
                        "win_rate": wr2, "win_rate_pct": fmt_pct(wr2),
                    })

    return pd.DataFrame(rows)


def pairwise_report(long_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for (tf, setup, rr), g in long_df.groupby(["timeframe", "setup", "rr"], dropna=False):
        # prepara bins por feature (quantis)
        bins = {}
        for feat in features:
            s = pd.to_numeric(g[feat], errors="coerce")
            if s.notna().sum() < MIN_TRADES:
                continue
            try:
                bins[feat] = pd.qcut(s, q=PAIRWISE_BINS, labels=False, duplicates="drop")
            except Exception:
                continue

        feats_ok = list(bins.keys())
        if len(feats_ok) < 2:
            continue

        for a, b in itertools.combinations(feats_ok, 2):
            ga = bins[a]
            gb = bins[b]
            tmp = g.copy()
            tmp["bin_a"] = ga
            tmp["bin_b"] = gb
            tmp = tmp.dropna(subset=["bin_a", "bin_b"])

            if tmp.shape[0] < MIN_TRADES:
                continue

            grouped = tmp.groupby(["bin_a", "bin_b"])
            for (ba, bb), sub in grouped:
                if sub.shape[0] < MIN_TRADES:
                    continue
                w = int((sub["outcome"] == "win").sum())
                l = int((sub["outcome"] == "loss").sum())
                d = w + l
                wr = w / d if d > 0 else np.nan

                rows.append({
                    "timeframe": tf, "setup": setup, "rr": rr,
                    "feature_a": a, "feature_b": b,
                    "bin_a": int(ba), "bin_b": int(bb),
                    "trades": int(sub.shape[0]), "wins": w, "losses": l,
                    "win_rate": wr, "win_rate_pct": fmt_pct(wr),
                })

    return pd.DataFrame(rows)


def best_patterns_md(uni: pd.DataFrame, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Best patterns — {SYMBOL}\n\n")
        f.write(f"- MIN_TRADES: {MIN_TRADES}\n")
        f.write(f"- Buckets: low/high {PCTS}\n\n")

        if uni.empty:
            f.write("Sem dados.\n")
            return

        for (tf, setup, rr), g in uni.groupby(["timeframe", "setup", "rr"]):
            g2 = g[g["bucket"] != "ALL"].copy()
            g2 = g2.dropna(subset=["win_rate"])
            g2 = g2.sort_values(["win_rate", "trades"], ascending=[False, False]).head(TOP_K_BEST)

            f.write(f"## {tf} | {setup} | RR {rr}\n\n")
            f.write("| Feature | Bucket | Trades | WinRate |\n")
            f.write("|---|---|---:|---:|\n")
            for _, r in g2.iterrows():
                f.write(f"| {r['feature']} | {r['bucket']} | {int(r['trades'])} | {r['win_rate_pct']} |\n")
            f.write("\n")


def main():
    os.makedirs("results", exist_ok=True)

    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError(f"Não achei {INPUT_PATH}. Rode o backtest primeiro.")

    trades = pd.read_csv(INPUT_PATH)

    long_df = melt_outcomes(trades)

    # features numéricas candidatas (remove colunas claramente não-features)
    exclude = set([
        "entry_price", "stop_price", "fill_delay",
        "timeframe", "setup", "side", "rr", "outcome",
    ])
    numeric_cols = []
    for c in long_df.columns:
        if c in exclude:
            continue
        if c in ["signal_time", "entry_time"]:
            continue
        # tenta converter e checa se é numérica
        s = pd.to_numeric(long_df[c], errors="coerce")
        if s.notna().sum() > 0:
            numeric_cols.append(c)

    # roda univariado e pairwise
    uni = univariate_report(long_df, numeric_cols)
    uni.to_csv(f"results/patterns_univariate_{SYMBOL}.csv", index=False)

    pair = pairwise_report(long_df, numeric_cols)
    pair.to_csv(f"results/patterns_pairwise_{SYMBOL}.csv", index=False)

    best_patterns_md(uni, f"results/patterns_best_{SYMBOL}.md")


if __name__ == "__main__":
    main()
