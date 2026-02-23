# src/stat.py
# Mixed-effects model (query as random intercept) + paired post-hoc within complexity
# PLUS: 6×3 two-way ANOVA table (strategy × complexity) for reporting.
#
# DVs: accuracy_10, coverage, quality_bertscore
# IVs: strategy (fixed), complexity (between-qid; fixed)
# Random effect: qid (random intercept)
#
# NOTE (paper grouping):
#   Collaborative = {hierarchical, peer_to_peer, functional}
#   Single-agent  = {dense, rag, human_proxy_rag}
#
# Run: python -m src.stat
from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy.stats import ttest_rel


# -----------------------
# 0) OUTPUT ROOT (outputs/eval/)
# -----------------------
EVAL_ROOT = Path("outputs/eval")
EVAL_ROOT.mkdir(parents=True, exist_ok=True)
STATS_DIR = EVAL_ROOT / "stats"


# -----------------------
# 1) FILES (one per strategy)
# -----------------------
FILES = [
    {"strategy": "rag", "path": "outputs/eval/rag_detailed.csv"},
    {"strategy": "dense", "path": "outputs/eval/denseretrieval_detailed.csv"},
    {"strategy": "hierarchical", "path": "outputs/eval/hierarchy_detailed.csv"},
    {"strategy": "peer_to_peer", "path": "outputs/eval/peer_detailed.csv"},
    {"strategy": "functional", "path": "outputs/eval/functional_detailed.csv"},
    {"strategy": "human_proxy_rag", "path": "outputs/eval/human_proxy_rag_detailed.csv"},
]

DATASET_FILTER: Optional[str] = None  # e.g., "trecdl2020"
METRICS = ["accuracy_10", "ndcg_10", "coverage", "quality_bertscore"]


# -----------------------
# 2) COLUMN NORMALIZATION
# -----------------------
COLUMN_ALIASES: Dict[str, List[str]] = {
    "qid": ["qid", "query_id", "queryid", "q_id"],
    "complexity": ["complexity", "query_complexity", "complexity_label"],
    "dataset": ["dataset", "corpus", "collection"],
    "accuracy_10": ["accuracy@10", "accuracy_10", "acc@10", "accuracy_at_10", "accuracy10"],
    "ndcg_10": ["ndcg@10", "ndcg_10", "ndcg10", "nDCG@10", "ndcg_at_10"],
    "coverage": ["coverage", "info_coverage", "information_coverage", "info cov.", "info_cov"],
    "quality_bertscore": ["quality_bertscore", "bertscore", "bert_score", "quality_bert"],
}

BASE_REQUIRED = ["qid", "complexity"]


def _normalize_header(h: str) -> str:
    h = h.strip().lower()
    h = re.sub(r"\s+", "_", h)
    h = h.replace("%", "pct")
    return h


def _read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".jsonl":
        return pd.read_json(path, lines=True)
    if suf == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {path}")


def _rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    orig_cols = list(df.columns)
    df = df.copy()
    df.columns = [_normalize_header(c) for c in df.columns]

    alias_to_canon: Dict[str, str] = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            alias_to_canon[_normalize_header(a)] = canon

    ren = {c: alias_to_canon[c] for c in df.columns if c in alias_to_canon}
    df = df.rename(columns=ren)

    missing_base = [c for c in BASE_REQUIRED if c not in df.columns]
    if missing_base:
        raise ValueError(
            f"Missing required base columns {missing_base}. Columns present: {orig_cols}"
        )

    for m in METRICS:
        if m not in df.columns:
            df[m] = np.nan

    keep = BASE_REQUIRED + METRICS
    if "dataset" in df.columns:
        keep.append("dataset")
    return df[keep]


def load_long_df(files: List[Dict[str, str]]) -> pd.DataFrame:
    all_rows = []
    for item in files:
        strat = item["strategy"]
        path = Path(item["path"])
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")

        df = _read_any(path)
        df = _rename_to_canonical(df)
        df["strategy"] = strat
        all_rows.append(df)

    long_df = pd.concat(all_rows, ignore_index=True)

    if DATASET_FILTER is not None:
        if "dataset" not in long_df.columns:
            raise ValueError("DATASET_FILTER is set but no 'dataset' column exists.")
        long_df = long_df[long_df["dataset"] == DATASET_FILTER].copy()

    long_df["qid"] = long_df["qid"].astype(str)
    long_df["strategy"] = long_df["strategy"].astype("category")
    long_df["complexity"] = long_df["complexity"].astype("category")

    for m in METRICS:
        long_df[m] = pd.to_numeric(long_df[m], errors="coerce")

    if long_df["coverage"].notna().any() and long_df["coverage"].max(skipna=True) > 1.0:
        long_df["coverage"] = long_df["coverage"] / 100.0

    return long_df


# -----------------------
# 3) TWO-WAY ANOVA (for reporting)
# -----------------------
def run_two_way_anova(df: pd.DataFrame, metric: str) -> Tuple[pd.DataFrame, str]:
    """
    Classical two-way ANOVA (Type II):
      metric ~ C(strategy) * C(complexity)

    NOTE: Treats rows as independent; keep for reporting only.
    """
    formula = f"{metric} ~ C(strategy) * C(complexity)"
    ols_fit = smf.ols(formula, data=df).fit()
    aov = anova_lm(ols_fit, typ=2)

    ss_res = aov.loc["Residual", "sum_sq"]
    aov["eta_p2"] = aov["sum_sq"] / (aov["sum_sq"] + ss_res)

    return aov, formula


# -----------------------
# 4) MIXED MODEL (kept as-is)
# -----------------------
def run_mixedlm(df: pd.DataFrame, metric: str):
    """
    Mixed-effects model:
      metric ~ C(strategy) * C(complexity)
      random intercept: qid
    """
    if isinstance(df["strategy"].dtype, pd.CategoricalDtype):
        df["strategy"] = df["strategy"].cat.remove_unused_categories()
    if isinstance(df["complexity"].dtype, pd.CategoricalDtype):
        df["complexity"] = df["complexity"].cat.remove_unused_categories()

    strat_levels = df["strategy"].nunique()
    cx_levels = df["complexity"].nunique()
    if strat_levels < 2:
        raise ValueError("Need >=2 strategy levels for mixed model.")

    if cx_levels < 2:
        formula = f"{metric} ~ C(strategy)"
    else:
        formula = f"{metric} ~ C(strategy) * C(complexity)"

    md = smf.mixedlm(formula, df, groups=df["qid"])
    res = md.fit(reml=False, method="lbfgs")
    return formula, res


# -----------------------
# 5) PAIRED post-hoc within complexity (qid-paired)  [replaces Tukey HSD]
# -----------------------
def _adjust_pvals(pvals: np.ndarray, method: str) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return pvals
    if method == "none":
        return pvals
    if method == "bonferroni":
        return np.minimum(1.0, pvals * m)

    # Holm step-down
    if method == "holm":
        order = np.argsort(pvals)
        ranked = pvals[order]
        adj = np.empty_like(ranked)
        for i, p in enumerate(ranked):
            adj[i] = min(1.0, (m - i) * p)
        for i in range(1, m):
            adj[i] = max(adj[i], adj[i - 1])  # monotone
        out = np.empty_like(adj)
        out[order] = adj
        return out

    raise ValueError(f"Unknown p_adjust method: {method}")


def run_paired_posthoc_within_complexity(
    df: pd.DataFrame,
    metric: str,
    *,
    alpha: float = 0.05,
    p_adjust: str = "holm",
) -> Dict[str, pd.DataFrame]:
    """
    Within each complexity, compare strategies using paired t-tests (paired by qid).

    Returns dict[cx] -> DataFrame:
      group1, group2, n_pairs, mean_diff, t, p_raw, p_adj, reject
    """
    out: Dict[str, pd.DataFrame] = {}

    for cx in df["complexity"].cat.categories:
        sub = df[df["complexity"] == cx].dropna(subset=[metric]).copy()
        if sub.empty or sub["strategy"].nunique() < 2:
            continue

        # qid × strategy matrix (paired by qid)
        mat = sub.pivot_table(index="qid", columns="strategy", values=metric, aggfunc="mean")
        strats = [c for c in mat.columns if mat[c].notna().any()]
        if len(strats) < 2:
            continue

        rows = []
        pvals = []

        for a, b in itertools.combinations(strats, 2):
            paired = mat[[a, b]].dropna()
            n = len(paired)
            if n < 2:
                continue

            x = paired[a].to_numpy(dtype=float)
            y = paired[b].to_numpy(dtype=float)

            t, p = ttest_rel(x, y, nan_policy="omit")
            mean_diff = float(np.mean(x - y))  # mean(group1 - group2)

            rows.append(
                {
                    "group1": str(a),
                    "group2": str(b),
                    "n_pairs": int(n),
                    "mean_diff": mean_diff,
                    "t": float(t),
                    "p_raw": float(p),
                }
            )
            pvals.append(float(p))

        if not rows:
            continue

        tbl = pd.DataFrame(rows)
        tbl["p_adj"] = _adjust_pvals(tbl["p_raw"].to_numpy(), p_adjust)
        tbl["reject"] = tbl["p_adj"] < alpha

        out[str(cx)] = tbl.sort_values(["p_adj", "p_raw"]).reset_index(drop=True)

    return out


# -----------------------
# 6) PAIRED effect size dz within complexity  [replaces pooled Cohen's d]
# -----------------------
SINGLE_STRATS = {"dense", "rag", "human_proxy_rag"}
COLLAB_STRATS = {"hierarchical", "peer_to_peer", "functional"}


def paired_cohens_dz(deltas: np.ndarray) -> float:
    """
    dz = mean(delta) / sd(delta), where delta is per-qid paired difference.
    """
    d = np.asarray(deltas, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return float("nan")
    sd = np.std(d, ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(np.mean(d) / sd)


def effect_sizes_within_complexity_dz(
    df: pd.DataFrame,
    metric: str,
    *,
    mode: str = "best_single",  # "best_single" or "mean_single"
) -> pd.DataFrame:
    """
    For each complexity:
      delta_qid = mean(collab) - best_single   (default)
                = mean(collab) - mean(single) (if mode="mean_single")
      dz = mean(delta)/sd(delta)
    """
    rows = []
    for cx in df["complexity"].cat.categories:
        sub = df[df["complexity"] == cx].dropna(subset=[metric]).copy()
        if sub.empty:
            continue

        mat = sub.pivot_table(index="qid", columns="strategy", values=metric, aggfunc="mean")

        collab_cols = [c for c in mat.columns if str(c) in COLLAB_STRATS]
        single_cols = [c for c in mat.columns if str(c) in SINGLE_STRATS]
        if not collab_cols or not single_cols:
            continue

        collab_q = mat[collab_cols].mean(axis=1, skipna=True)

        if mode == "best_single":
            single_q = mat[single_cols].max(axis=1, skipna=True)
        elif mode == "mean_single":
            single_q = mat[single_cols].mean(axis=1, skipna=True)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        delta = (collab_q - single_q).dropna()

        rows.append(
            {
                "complexity": str(cx),
                "n_qids": int(delta.shape[0]),
                "mean_delta": float(delta.mean()) if delta.shape[0] else float("nan"),
                "sd_delta": float(delta.std(ddof=1)) if delta.shape[0] > 1 else float("nan"),
                "cohens_dz": paired_cohens_dz(delta.to_numpy()),
                "single_mode": mode,
            }
        )

    return pd.DataFrame(rows)


# -----------------------
# 7) SYNERGY (per-qid)  [unchanged]
# -----------------------
SINGLE_ONLY_STRATS = ["dense", "rag", "human_proxy_rag"]
COLLAB_ONLY_STRATS = ["hierarchical", "peer_to_peer", "functional"]


def compute_synergy_per_qid(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    single = df[df["strategy"].isin(SINGLE_ONLY_STRATS)].copy()
    collab = df[df["strategy"].isin(COLLAB_ONLY_STRATS)].copy()

    single = single.dropna(subset=[metric])
    collab = collab.dropna(subset=[metric])

    if single.empty or collab.empty:
        return pd.DataFrame(
            columns=[
                "qid",
                "complexity",
                "collab_strategy",
                "metric",
                "collab_score",
                "max_single_score",
                "synergy",
            ]
        )

    max_single = (
        single.groupby(["qid", "complexity"], as_index=False)[metric]
        .max()
        .rename(columns={metric: "max_single_score"})
    )

    collab = collab.rename(columns={metric: "collab_score"})
    collab = collab.rename(columns={"strategy": "collab_strategy"})

    out = collab.merge(max_single, on=["qid", "complexity"], how="inner")
    out["metric"] = metric
    out["synergy"] = out["collab_score"] - out["max_single_score"]

    return out[
        [
            "qid",
            "complexity",
            "collab_strategy",
            "metric",
            "collab_score",
            "max_single_score",
            "synergy",
        ]
    ]


def summarize_synergy(syn: pd.DataFrame) -> pd.DataFrame:
    if syn.empty:
        return pd.DataFrame(
            columns=[
                "collab_strategy",
                "complexity",
                "n",
                "mean_synergy",
                "median_synergy",
                "pct_positive",
            ]
        )

    g = syn.groupby(["collab_strategy", "complexity"], as_index=False)["synergy"]
    summ = g.agg(
        n="count",
        mean_synergy="mean",
        median_synergy="median",
    )
    pos = (
        syn.assign(pos=(syn["synergy"] > 0).astype(int))
        .groupby(["collab_strategy", "complexity"], as_index=False)["pos"]
        .mean()
        .rename(columns={"pos": "pct_positive"})
    )
    summ = summ.merge(pos, on=["collab_strategy", "complexity"], how="left")
    summ["pct_positive"] = 100.0 * summ["pct_positive"]
    return summ


def main():
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    long_df = load_long_df(FILES)

    out_long = EVAL_ROOT / "stacked_all_strategies_long.csv"
    long_df.to_csv(out_long, index=False)
    print(f"Saved stacked DF: {out_long.resolve()}")

    print("\nCounts per (strategy, complexity):")
    print(long_df.groupby(["strategy", "complexity"], observed=False).size().unstack(fill_value=0))

    for metric in METRICS:
        print("\n" + "=" * 90)
        print(f"METRIC: {metric}")

        dfm = long_df.dropna(subset=[metric]).copy()
        if dfm.empty:
            print(f"[SKIP] No non-NaN rows for metric={metric}")
            continue

        if isinstance(dfm["strategy"].dtype, pd.CategoricalDtype):
            dfm["strategy"] = dfm["strategy"].cat.remove_unused_categories()
        if isinstance(dfm["complexity"].dtype, pd.CategoricalDtype):
            dfm["complexity"] = dfm["complexity"].cat.remove_unused_categories()

        print("Strategies with this metric:", sorted(dfm["strategy"].unique().tolist()))
        print("QIDs:", dfm["qid"].nunique(), "Rows:", len(dfm))

        # -------------------------
        # (A) TWO-WAY ANOVA (reporting)
        # -------------------------
        try:
            aov_tbl, aov_formula = run_two_way_anova(dfm, metric)
            print("\n--- Two-way ANOVA (Type II; reporting) ---")
            print("Formula:", aov_formula)
            print(aov_tbl)

            aov_path = STATS_DIR / f"anova_{metric}.csv"
            aov_tbl.to_csv(aov_path)
            print(f"Saved ANOVA: {aov_path.resolve()}")
        except Exception as e:
            print(f"[WARN] ANOVA failed for metric={metric}: {e}")

        # -------------------------
        # (B) Mixed model (kept as-is)
        # -------------------------
        try:
            formula, res = run_mixedlm(dfm, metric)
        except Exception as e:
            print(f"[SKIP] MixedLM failed for metric={metric}: {e}")
            res = None

        if res is not None:
            print("\n--- Mixed Effects Model (kept as-is) ---")
            print("Formula:", formula)
            print(res.summary())

            fe = pd.DataFrame(
                {
                    "coef": res.fe_params,
                    "se": res.bse_fe,
                    "z": res.tvalues[res.fe_params.index],
                    "p": res.pvalues[res.fe_params.index],
                }
            )
            fe_path = STATS_DIR / f"mixedlm_fixed_effects_{metric}.csv"
            fe.to_csv(fe_path)
            print(f"Saved fixed effects: {fe_path.resolve()}")

        # -------------------------
        # (C) Paired post-hoc within complexity (replaces Tukey HSD)
        # -------------------------
        print("\n--- Paired post-hoc within each complexity (qid-paired; Holm-adjusted) ---")
        post = run_paired_posthoc_within_complexity(dfm, metric, alpha=0.05, p_adjust="holm")
        for cx, tbl in post.items():
            print(f"\n[complexity = {cx}]")
            print(tbl)
            outp = STATS_DIR / f"paired_posthoc_{metric}_{cx}.csv"
            tbl.to_csv(outp, index=False)

        # -------------------------
        # (D) Effect sizes dz within complexity (paired; replaces pooled Cohen's d)
        # -------------------------
        print("\n--- Effect sizes (paired Cohen's dz) within each complexity ---")
        es = effect_sizes_within_complexity_dz(dfm, metric, mode="best_single")
        if not es.empty:
            print(es.to_string(index=False))
            es_path = STATS_DIR / f"effect_sizes_{metric}_dz_by_complexity.csv"
            es.to_csv(es_path, index=False)
            print(f"Saved effect sizes: {es_path.resolve()}")
        else:
            print("[SKIP] No effect sizes computed (insufficient data).")

        # -------------------------
        # (E) Synergy (per-qid) for accuracy_10 and coverage
        # -------------------------
        if metric in {"accuracy_10", "ndcg_10", "coverage"}:
            syn = compute_synergy_per_qid(dfm, metric)
            syn_path = STATS_DIR / f"synergy_{metric}_per_qid.csv"
            syn.to_csv(syn_path, index=False)
            print(f"\nSaved synergy per-qid: {syn_path.resolve()}")

            summ = summarize_synergy(syn)
            summ_path = STATS_DIR / f"synergy_{metric}_summary_by_complexity.csv"
            summ.to_csv(summ_path, index=False)

            print("\n--- Synergy summary (by collab_strategy × complexity) ---")
            if not summ.empty:
                print(summ.to_string(index=False))
            else:
                print("[SKIP] No synergy computed (no overlap between collab and single-agent for this metric).")

            print(f"Saved synergy summary: {summ_path.resolve()}")

    print("\nSaved outputs to:", STATS_DIR.resolve())


if __name__ == "__main__":
    main()
