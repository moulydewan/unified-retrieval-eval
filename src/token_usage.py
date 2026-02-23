import json
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("outputs/results")
OUT_PATH = Path("outputs/eval/token_time_by_strategy_dataset.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

ALLOWED_STRATEGIES = {
    "rag",
    "denseretrieval",
    "human_proxy_rag",
    "hierarchy_agents",
    "peer_agents",
    "functional_agents",
}

# Add BEIR / SCIDOCS / NQ (adjust keys to match your filenames)
ALLOWED_DATASETS = {
    "trecdl_2019",
    "trecdl_2020",
    "trecdl_2021",
    "beir",
    "nq",
}


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_meta_from_filename(path: Path):
    # model__strategy__dataset__*.jsonl
    parts = path.stem.split("__")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {path.name}")

    dataset = parts[2]

    # ---- NORMALIZE DATASET KEYS ----
    if dataset in {"beir", "beir_scidocs"}:
        dataset = "beir_scidocs"
    if dataset in {"naturalquestions", "natural_question"}:
        dataset = "nq"

    return {
        "model": parts[0],
        "strategy": parts[1],
        "dataset": dataset,
        "file": path.name,
    }


def extract_tokens(row):
    # Peer/Functional: usage_total ; others: usage
    usage = row.get("usage_total") or row.get("usage")
    if not usage:
        return None
    return {
        "input": int(usage.get("inputTokens", 0) or 0),
        "output": int(usage.get("outputTokens", 0) or 0),
        "total": int(usage.get("totalTokens", 0) or 0),
    }


def main():
    token_rows = []
    run_rows = []

    jsonl_files = sorted(RESULTS_DIR.glob("*.jsonl"))
    if not jsonl_files:
        raise RuntimeError(f"No JSONL files found in {RESULTS_DIR}")

    for path in jsonl_files:
        meta = parse_meta_from_filename(path)

        if meta["strategy"] not in ALLOWED_STRATEGIES:
            continue
        if meta["dataset"] not in ALLOWED_DATASETS:
            continue

        rows = list(load_jsonl(path))
        if not rows:
            continue

        # --- total run time (one per file) ---
        run_time_sec = None
        for r in rows:
            t = r.get("experiment_time_sec")
            if t is not None:
                try:
                    run_time_sec = float(t)
                    break
                except Exception:
                    pass

        run_rows.append(
            {
                "strategy": meta["strategy"],
                "dataset": meta["dataset"],
                "model": meta["model"],
                "file": meta["file"],
                "run_time_sec": run_time_sec,
                "n_queries": len(rows),
            }
        )

        # --- per-query tokens ---
        for r in rows:
            tok = extract_tokens(r)
            if not tok:
                continue
            token_rows.append(
                {
                    "strategy": meta["strategy"],
                    "dataset": meta["dataset"],
                    "model": meta["model"],
                    "input_tokens": tok["input"],
                    "output_tokens": tok["output"],
                    "total_tokens": tok["total"],
                }
            )

    df_tokens = pd.DataFrame(token_rows)
    df_runs = pd.DataFrame(run_rows)

    if df_tokens.empty:
        raise RuntimeError("No token data found (usage / usage_total missing?).")
    if df_runs.empty:
        raise RuntimeError("No runs found after filtering.")

    # -----------------------------
    # Dataset-wise aggregation
    # -----------------------------

    # --- tokens per (strategy, dataset) ---
    tok_by_sd = (
        df_tokens.groupby(["strategy", "dataset"])
        .agg(
            n_queries_with_tokens=("total_tokens", "count"),
            total_input_tokens=("input_tokens", "sum"),
            total_output_tokens=("output_tokens", "sum"),
            total_tokens=("total_tokens", "sum"),
            avg_tokens_per_query=("total_tokens", "mean"),
        )
        .reset_index()
    )

    # --- runtime per (strategy, dataset) ---
    # time is run-level; sum over files, NOT over rows.
    time_by_sd = (
        df_runs.groupby(["strategy", "dataset"])
        .agg(
            n_runs=("file", "count"),
            total_run_time_sec=("run_time_sec", "sum"),
            avg_run_time_sec=("run_time_sec", "mean"),
            total_queries=("n_queries", "sum"),
        )
        .reset_index()
    )
    time_by_sd["avg_time_per_query_sec"] = (
        time_by_sd["total_run_time_sec"] / time_by_sd["total_queries"]
    )

    out = tok_by_sd.merge(time_by_sd, on=["strategy", "dataset"], how="left")

    # Helpful: share of missing tokens (sanity check)
    out["token_coverage_frac"] = out["n_queries_with_tokens"] / out["total_queries"]

    out = out.sort_values(["dataset", "strategy"])

    out.to_csv(OUT_PATH, index=False)
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
