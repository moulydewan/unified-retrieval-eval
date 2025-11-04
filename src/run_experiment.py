import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from src.utils import (
    setup_logging, load_config, save_jsonl,
    create_run_id, ExperimentTracker
)
from src.datasets.trecdl import TRECDLAdapter
from src.experiments.rag import retrieve_bm25, rag_synthesis

# ---------------------------
# Paths
# ---------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "configs"
OUTPUT_DIR = ROOT_DIR / "outputs"

logger = None


# ---------------------------
# Run a Single Experiment
# ---------------------------
def run_single_experiment(
    model_cfg: Dict[str, Any],
    strategy_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    dataset_name: str,
    strategy_name: str,
    model_name: str,
    limit: int,
    top_k: int,
    year: int | None = None,
) -> str:
    """Run a single (dataset + strategy + model) experiment."""
    start_time = time.time()

    # determine year (CLI > config > 2020 default)
    chosen_year = year if year is not None else dataset_cfg.get("year", 2020)

    logger.info(f"▶ Running {model_name} + {strategy_name} + {dataset_name} "
                f"(year={chosen_year}, limit={limit}, top_k={top_k})")

    # ===== Dataset Load =====
    adapter = TRECDLAdapter(
        year=chosen_year,
        mode=dataset_cfg.get("mode", "passage")
    )
    raw = adapter.load(limit=limit)
    df = adapter.trec_df(raw)
    logger.info(f"Loaded TREC DL {chosen_year} data: {df.shape[0]} rows")

    # ===== BM25 Retrieval =====
    bm25_df = retrieve_bm25(df, top_k=top_k)
    logger.info(f"BM25 retrieved {len(bm25_df)} passages (top_k={top_k})")

    # ===== RAG Synthesis with progress =====
    logger.info(f"Generating synthesized answers with {model_name}...")
    answers = []
    unique_queries = bm25_df[["qid", "query"]].drop_duplicates()

    for _, row in tqdm(unique_queries.iterrows(), total=len(unique_queries),
                       desc=f"{model_name}-{strategy_name}-{dataset_name}-{chosen_year}"):
        qid, query = row["qid"], row["query"]

        try:
            result = rag_synthesis(
                bm25_df=bm25_df[bm25_df["qid"] == qid],
                model_id=model_cfg["model_id"],
                temperature=strategy_cfg.get("temperature", 0.7),
                max_tokens=model_cfg.get("max_tokens", 512),
                top_k=top_k,
            )
            answers.extend(result)
        except Exception as e:
            logger.error(f"Generation failed for qid={qid}: {e}")

    # ===== Save Results =====
    run_id = create_run_id(model_name, strategy_name, f"{dataset_name}{chosen_year}")
    output_path = OUTPUT_DIR / "results" / f"{run_id}.jsonl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_jsonl(answers, output_path)

    elapsed = time.time() - start_time
    logger.info(f"Saved results → {output_path}")
    logger.info(f"Experiment completed in {elapsed:.2f}s "
                f"({len(answers)} answers generated).")

    return str(output_path)


# ---------------------------
# Main CLI Entrypoint
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Run IR experiments (TREC DL + RAG + Claude)")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated list of model names")
    parser.add_argument("--strategies", type=str, required=True, help="Comma-separated list of strategy names")
    parser.add_argument("--datasets", type=str, required=True, help="Comma-separated list of dataset names")
    parser.add_argument("--year", type=int, default=None, help="Dataset year (e.g., 2019, 2020, 2021). Default: 2020")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of queries (default=10)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k documents to retrieve (default=20 or from strategy)")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional custom output dir")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    args = parser.parse_args()

    # ----- Setup Output Dir -----
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Setup Logging -----
    global logger
    logger = setup_logging(OUTPUT_DIR / "logs", "run_experiment")

    # ----- Load Configs -----
    datasets_cfg = load_config(CONFIG_DIR / "datasets.yaml")["datasets"]
    models_cfg = load_config(CONFIG_DIR / "models.yaml")["models"]
    strategies_cfg = load_config(CONFIG_DIR / "strategies.yaml")["strategies"]

    # ----- Parse Input -----
    dataset_names = [d.strip() for d in args.datasets.split(",")]
    model_names = [m.strip() for m in args.models.split(",")]
    strategy_names = [s.strip() for s in args.strategies.split(",")]

    # ----- Validation Helpers -----
    def get_model(name):
        for m in models_cfg:
            if m["name"] == name:
                return m
        raise ValueError(f"Model '{name}' not found in models.yaml")

    def get_strategy(name):
        if name in strategies_cfg:
            return strategies_cfg[name]
        raise ValueError(f"Strategy '{name}' not found in strategies.yaml")

    def get_dataset(name):
        if name in datasets_cfg:
            return datasets_cfg[name]
        raise ValueError(f"Dataset '{name}' not found in datasets.yaml")

    # ----- Build Plan -----
    experiments = []
    for dataset_name in dataset_names:
        for strategy_name in strategy_names:
            for model_name in model_names:
                experiments.append({
                    "dataset_name": dataset_name,
                    "strategy_name": strategy_name,
                    "model_name": model_name,
                    "dataset_cfg": get_dataset(dataset_name),
                    "strategy_cfg": get_strategy(strategy_name),
                    "model_cfg": get_model(model_name)
                })

    logger.info(f"Planned {len(experiments)} experiment(s):")
    for i, e in enumerate(experiments, 1):
        logger.info(f"  {i}. {e['dataset_name']} + {e['strategy_name']} + {e['model_name']}")

    if args.dry_run:
        logger.info("Dry run only. Exiting.")
        return

    # ----- Run Experiments -----
    tracker = ExperimentTracker(OUTPUT_DIR)
    total_start = time.time()

    for i, exp in enumerate(experiments, 1):
        logger.info(f"\nStarting experiment {i}/{len(experiments)}")
        exp_start = time.time()
        try:
            result_path = run_single_experiment(
                model_cfg=exp["model_cfg"],
                strategy_cfg=exp["strategy_cfg"],
                dataset_cfg=exp["dataset_cfg"],
                dataset_name=exp["dataset_name"],
                strategy_name=exp["strategy_name"],
                model_name=exp["model_name"],
                limit=args.limit,
                top_k=args.top_k or exp["strategy_cfg"].get("top_k", 20),
                year=args.year,
            )
            exp_time = time.time() - exp_start
            tracker.add_run({
                "run_id": Path(result_path).stem,
                "dataset": exp["dataset_name"],
                "year": args.year,
                "strategy": exp["strategy_name"],
                "model": exp["model_name"],
                "limit": args.limit,
                "top_k": args.top_k,
                "output_path": str(result_path),
                "duration_sec": round(exp_time, 2)
            })
            logger.info(f"Completed {exp['model_name']} + {exp['strategy_name']} "
                        f"in {exp_time:.2f}s\n")
        except Exception as e:
            logger.error(f"Experiment {i} failed: {e}")
            continue

    total_time = time.time() - total_start
    logger.info(f"🏁 All experiments done in {total_time:.2f}s.")
    logger.info(f"Metadata saved → {OUTPUT_DIR / 'experiment_metadata.json'}")


if __name__ == "__main__":
    main()
