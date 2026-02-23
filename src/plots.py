# from pathlib import Path
# from matplotlib.lines import Line2D
# import matplotlib.pyplot as plt
# import pandas as pd

# plt.rcParams.update({
#     "font.size": 12,
#     "axes.titlesize": 14,
#     "axes.labelsize": 13,
#     "xtick.labelsize": 11,
#     "ytick.labelsize": 11,
#     "legend.fontsize": 11,
# })


# IN_PATH = Path("outputs/eval/cost_analysis.csv")
# OUT_DIR = Path("outputs/eval/plots")
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# # Optional: nicer display names in the plot
# STRATEGY_LABELS = {
#     "rag": "RAG",
#     "human_proxy_rag": "Human Proxy RAG",
#     "hierarchy_agents": "Hierarchical",
#     "peer_agents": "Peer-to-Peer",
#     "functional_agents": "Functional",
#     "denseretrieval": "Dense Retrieval",
# }

# DATASET_LABELS = {
#     "trecdl_2019": "TREC DL'19",
#     "trecdl_2020": "TREC DL'20",
#     "trecdl_2021": "TREC DL'21",
#     "beir_scidocs": "BEIR / SCIDOCS",
#     "nq": "Natural Questions",
# }

# # Bubble sizing
# BUBBLE_MIN = 60
# BUBBLE_MAX = 800


# def _read_cost_benefit_csv(path: Path) -> pd.DataFrame:
#     # Robust CSV read: handle comma/semicolon delimiters and thousands separators.
#     # Using engine="python" lets pandas auto-detect delimiter more reliably.
#     df = pd.read_csv(path, thousands=",", sep=None, engine="python", encoding="utf-8-sig")

#     df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

#     # strip "2. rag" -> "rag"
#     if "strategy" in df.columns:
#         df["strategy"] = (
#             df["strategy"]
#             .astype(str)
#             .str.strip()
#             .str.replace(r"^\s*\d+\.\s*", "", regex=True)
#         )

#     if "dataset" in df.columns:
#         df["dataset"] = df["dataset"].astype(str).str.strip()

#     # Your columns might be named ndcg@10 or ndcg_10 depending on export—support both
#     if "ndcg@10" not in df.columns and "ndcg_10" in df.columns:
#         df = df.rename(columns={"ndcg_10": "ndcg@10"})

#     if "Coverage" not in df.columns and "coverage" in df.columns:
#         df = df.rename(columns={"coverage": "Coverage"})

#     for c in ["avg_tokens_per_query", "ndcg@10", "Coverage"]:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce")

#     return df



# def _scale_bubbles(series: pd.Series) -> pd.Series:
#     """
#     Map Coverage in [min,max] -> bubble areas in [BUBBLE_MIN, BUBBLE_MAX].
#     If Coverage missing or constant, fall back to a constant size.
#     """
#     s = series.copy()
#     if s.isna().all():
#         return pd.Series([200] * len(s), index=s.index)

#     s_min = s.min(skipna=True)
#     s_max = s.max(skipna=True)
#     if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
#         return pd.Series([200] * len(s), index=s.index)

#     return BUBBLE_MIN + (s - s_min) * (BUBBLE_MAX - BUBBLE_MIN) / (s_max - s_min)


# def _label_offset(strategy: str, dataset: str):
#     base = (6, 4)
#     ds_nudge = {
#         "trecdl_2019": (0, 10),
#         "trecdl_2020": (0, 0),
#         "trecdl_2021": (0, -10),
#     }.get(dataset, (0, 0))
#     return (base[0] + ds_nudge[0], base[1] + ds_nudge[1])


# def plot_cost_vs_metric_combined(
#     df: pd.DataFrame,
#     y_col: str,
#     y_label: str,
#     out_name: str,
#     title: str,
# ) -> None:
#     required = {"strategy", "dataset", "avg_tokens_per_query", y_col}
#     missing = required - set(df.columns)
#     if missing:
#         raise ValueError(f"Missing required columns: {sorted(missing)}")

#     ds_order = ["trecdl_2019", "trecdl_2020", "trecdl_2021"]
#     dff = df[df["dataset"].isin(ds_order)].copy()

#     ds_marker = {"trecdl_2019": "o", "trecdl_2020": "^", "trecdl_2021": "s"}
#     ds_color = {"trecdl_2019": "C0", "trecdl_2020": "C1", "trecdl_2021": "C2"}

#     fig, ax = plt.subplots(figsize=(6.8, 5.8))
#     MARKER_SIZE = 120

#     # --- padding ---
#     x = dff["avg_tokens_per_query"].astype(float)
#     y = dff[y_col].astype(float)

#     xpad = (x.max() - x.min()) * 0.05 if x.max() > x.min() else 100
#     ypad = (y.max() - y.min()) * 0.08 if y.max() > y.min() else 0.05
#     ax.set_xlim(x.min() - xpad, x.max() + xpad)
#     ax.set_ylim(y.min() - ypad, y.max() + ypad)

#     # plot points + labels
#     for ds in ds_order:
#         dsd = dff[dff["dataset"] == ds]
#         if dsd.empty:
#             continue

#         ax.scatter(
#             dsd["avg_tokens_per_query"],
#             dsd[y_col],
#             s=MARKER_SIZE,
#             marker=ds_marker[ds],
#             alpha=0.85,
#             color=ds_color[ds],
#         )

#         for _, r in dsd.iterrows():
#             strat = r["strategy"]
#             label = STRATEGY_LABELS.get(strat, strat)

#             dx, dy = _label_offset(strat, ds)

#             xmin, xmax = ax.get_xlim()
#             ymin, ymax = ax.get_ylim()
#             xr = float(r["avg_tokens_per_query"])
#             yr = float(r[y_col])

#             if xr > xmax - 0.08 * (xmax - xmin):
#                 dx = -abs(dx) - 6
#             if yr > ymax - 0.08 * (ymax - ymin):
#                 dy = -abs(dy) - 4

#             ax.annotate(
#                 label,
#                 (xr, yr),
#                 textcoords="offset points",
#                 xytext=(dx, dy),
#                 ha="left" if dx >= 0 else "right",
#                 va="bottom" if dy >= 0 else "top",
#                 fontsize=11,
#                 fontweight="medium",
#                 clip_on=True,
#             )

#     ax.set_title(title)
#     ax.set_xlabel("Avg tokens per query (cost)")
#     ax.set_ylabel(y_label)
#     ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

#     legend_handles = [
#         Line2D([0], [0],
#                marker=ds_marker[ds],
#                color="w",
#                label=DATASET_LABELS.get(ds, ds),
#                markerfacecolor=ds_color[ds],
#                markersize=9)
#         for ds in ds_order
#         if ds in dff["dataset"].unique()
#     ]
#     ax.legend(handles=legend_handles, title="Dataset", loc="lower right", frameon=True)

#     out_path = OUT_DIR / out_name
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=300)
#     plt.close(fig)
#     print(f"Saved: {out_path}")


# def main():
#     df = _read_cost_benefit_csv(IN_PATH)

#     # Plot 1: Cost vs nDCG
#     df_ndcg = df.dropna(subset=["avg_tokens_per_query", "ndcg@10"])
#     plot_cost_vs_metric_combined(
#         df_ndcg,
#         y_col="ndcg@10",
#         y_label="nDCG@10 (retrieval quality)",
#         out_name="cost_vs_ndcg__trecdl_combined.png",
#         title="Performance vs Computational Cost Across TREC DL Benchmarks",
#     )

#     # Plot 2: Cost vs Coverage (drop rows where Coverage is missing, e.g., Dense Retrieval)
#     df_cov = df.dropna(subset=["avg_tokens_per_query", "Coverage"])
#     plot_cost_vs_metric_combined(
#         df_cov,
#         y_col="Coverage",
#         y_label="Coverage (generation quality)",
#         out_name="cost_vs_coverage__trecdl_combined.png",
#         title="Coverage vs Computational Cost Across TREC DL Benchmarks",
#     )


# if __name__ == "__main__":
#     main()




from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})


IN_PATH = Path("outputs/eval/cost_analysis.csv")
OUT_DIR = Path("outputs/eval/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: nicer display names in the plot
STRATEGY_LABELS = {
    "rag": "RAG",
    "human_proxy_rag": "Human Proxy RAG",
    "hierarchy_agents": "Hierarchical",
    "peer_agents": "Peer-to-Peer",
    "functional_agents": "Functional",
    "denseretrieval": "Dense Retrieval",
}

DATASET_LABELS = {
    "trecdl_2019": "TREC DL'19",
    "trecdl_2020": "TREC DL'20",
    "trecdl_2021": "TREC DL'21",
    "beir_scidocs": "BEIR / SCIDOCS",
    "nq": "Natural Questions",
}

# Bubble sizing
BUBBLE_MIN = 60
BUBBLE_MAX = 800


def _read_cost_benefit_csv(path: Path) -> pd.DataFrame:
    # Robust CSV read: handle comma/semicolon delimiters and thousands separators.
    # Using engine="python" lets pandas auto-detect delimiter more reliably.
    df = pd.read_csv(path, thousands=",", sep=None, engine="python", encoding="utf-8-sig")

    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    # strip "2. rag" -> "rag"
    if "strategy" in df.columns:
        df["strategy"] = (
            df["strategy"]
            .astype(str)
            .str.strip()
            .str.replace(r"^\s*\d+\.\s*", "", regex=True)
        )

    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].astype(str).str.strip()

    # Your columns might be named ndcg@10 or ndcg_10 depending on export—support both
    if "ndcg@10" not in df.columns and "ndcg_10" in df.columns:
        df = df.rename(columns={"ndcg_10": "ndcg@10"})

    if "Coverage" not in df.columns and "coverage" in df.columns:
        df = df.rename(columns={"coverage": "Coverage"})

    for c in ["avg_tokens_per_query", "ndcg@10", "Coverage"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df



def _scale_bubbles(series: pd.Series) -> pd.Series:
    """
    Map Coverage in [min,max] -> bubble areas in [BUBBLE_MIN, BUBBLE_MAX].
    If Coverage missing or constant, fall back to a constant size.
    """
    s = series.copy()
    if s.isna().all():
        return pd.Series([200] * len(s), index=s.index)

    s_min = s.min(skipna=True)
    s_max = s.max(skipna=True)
    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        return pd.Series([200] * len(s), index=s.index)

    return BUBBLE_MIN + (s - s_min) * (BUBBLE_MAX - BUBBLE_MIN) / (s_max - s_min)


def _label_offset(strategy: str, dataset: str):
    base = (6, 4)
    ds_nudge = {
        "trecdl_2019": (0, 10),
        "trecdl_2020": (0, 0),
        "trecdl_2021": (0, -10),
    }.get(dataset, (0, 0))
    return (base[0] + ds_nudge[0], base[1] + ds_nudge[1])

def _overlaps(b1, b2, pad=2.0) -> bool:
    """Return True if two display-coordinate bboxes overlap (with padding)."""
    return not (
        (b1.x1 + pad) < b2.x0 or
        (b1.x0 - pad) > b2.x1 or
        (b1.y1 + pad) < b2.y0 or
        (b1.y0 - pad) > b2.y1
    )


def _annotate_no_overlap(
    ax,
    text: str,
    xy: tuple,
    base_xytext: tuple,
    ha: str,
    va: str,
    placed_bboxes: list,
    max_tries: int = 30,
    step: int = 8,
    fontsize: int = 11,
):
    """
    Place an annotation, nudging it in y (and a tiny x) until it doesn't overlap
    any previously placed label bboxes.
    """
    fig = ax.figure
    dx0, dy0 = base_xytext

    # Try a sequence of y nudges: 0, +step, -step, +2step, -2step, ...
    nudges = [0]
    for k in range(1, max_tries):
        nudges.append(((k + 1) // 2) * step * (1 if k % 2 == 1 else -1))

    ann = None
    for k, dy_nudge in enumerate(nudges):
        # small x drift as tries increase (helps when many labels stack)
        dx_nudge = (k // 6) * 4

        if ann is not None:
            ann.remove()

        ann = ax.annotate(
            text,
            xy,
            textcoords="offset points",
            xytext=(dx0 + dx_nudge, dy0 + dy_nudge),
            ha=ha,
            va=va,
            fontsize=fontsize,
            fontweight="medium",
            clip_on=True,
        )

        # Need a draw to get accurate bbox
        fig.canvas.draw()
        bbox = ann.get_window_extent(renderer=fig.canvas.get_renderer())

        if not any(_overlaps(bbox, b) for b in placed_bboxes):
            placed_bboxes.append(bbox)
            return ann

    # fallback: accept last placement
    fig.canvas.draw()
    bbox = ann.get_window_extent(renderer=fig.canvas.get_renderer())
    placed_bboxes.append(bbox)
    return ann



def plot_cost_vs_metric_combined(
    df: pd.DataFrame,
    y_col: str,
    y_label: str,
    out_name: str,
    title: str,
) -> None:
    required = {"strategy", "dataset", "avg_tokens_per_query", y_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    ds_order = ["trecdl_2019", "trecdl_2020", "trecdl_2021"]
    dff = df[df["dataset"].isin(ds_order)].copy()

    ds_marker = {"trecdl_2019": "o", "trecdl_2020": "^", "trecdl_2021": "s"}
    ds_color = {"trecdl_2019": "C0", "trecdl_2020": "C1", "trecdl_2021": "C2"}

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    MARKER_SIZE = 120

    placed_bboxes = []

    # --- padding ---
    x = dff["avg_tokens_per_query"].astype(float)
    y = dff[y_col].astype(float)

    xpad = (x.max() - x.min()) * 0.05 if x.max() > x.min() else 100
    ypad = (y.max() - y.min()) * 0.08 if y.max() > y.min() else 0.05
    ax.set_xlim(x.min() - xpad, x.max() + xpad)
    ax.set_ylim(y.min() - ypad, y.max() + ypad)

    # plot points + labels
    for ds in ds_order:
        dsd = dff[dff["dataset"] == ds]
        if dsd.empty:
            continue

        ax.scatter(
            dsd["avg_tokens_per_query"],
            dsd[y_col],
            s=MARKER_SIZE,
            marker=ds_marker[ds],
            alpha=0.85,
            color=ds_color[ds],
        )

        for _, r in dsd.sort_values(by=y_col, ascending=False).iterrows():
            strat = r["strategy"]
            label = STRATEGY_LABELS.get(strat, strat)

            dx, dy = _label_offset(strat, ds)

            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            xr = float(r["avg_tokens_per_query"])
            yr = float(r[y_col])

            if xr > xmax - 0.08 * (xmax - xmin):
                dx = -abs(dx) - 6
            if yr > ymax - 0.08 * (ymax - ymin):
                dy = -abs(dy) - 4

            ha = "left" if dx >= 0 else "right"
            va = "bottom" if dy >= 0 else "top"

            _annotate_no_overlap(
                ax=ax,
                text=label,
                xy=(xr, yr),
                base_xytext=(dx, dy),
                ha=ha,
                va=va,
                placed_bboxes=placed_bboxes,
                max_tries=30,
                step=8,
                fontsize=11,
            )

    ax.set_title(title)
    ax.set_xlabel("Avg tokens per query (cost)")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    legend_handles = [
        Line2D([0], [0],
               marker=ds_marker[ds],
               color="w",
               label=DATASET_LABELS.get(ds, ds),
               markerfacecolor=ds_color[ds],
               markersize=9)
        for ds in ds_order
        if ds in dff["dataset"].unique()
    ]
    ax.legend(handles=legend_handles, title="Dataset", loc="lower right", frameon=True)

    out_path = OUT_DIR / out_name
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    df = _read_cost_benefit_csv(IN_PATH)

    # # Plot 1: Cost vs nDCG
    # df_ndcg = df.dropna(subset=["avg_tokens_per_query", "ndcg@10"])
    # plot_cost_vs_metric_combined(
    #     df_ndcg,
    #     y_col="ndcg@10",
    #     y_label="nDCG@10 (retrieval quality)",
    #     out_name="cost_vs_ndcg__trecdl_combined.png",
    #     title="Performance vs Computational Cost Across TREC DL Benchmarks",
    # )

    # Plot 2: Cost vs Coverage (drop rows where Coverage is missing, e.g., Dense Retrieval)
    df_cov = df.dropna(subset=["avg_tokens_per_query", "Coverage"])
    plot_cost_vs_metric_combined(
        df_cov,
        y_col="Coverage",
        y_label="Coverage (generation quality)",
        out_name="cost_vs_coverage__trecdl_combined.png",
        title="Coverage vs Computational Cost Across TREC DL Benchmarks",
    )


if __name__ == "__main__":
    main()
