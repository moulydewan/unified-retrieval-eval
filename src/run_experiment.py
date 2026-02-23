# src/run_experiment.py
import argparse
import time
from pathlib import Path
from typing import Dict, Any

from src.utils import (
    setup_logging, load_config, save_jsonl,
    create_run_id, ExperimentTracker
)
from src.managers.dataset_manager import DatasetManager
from src.managers.strategy_manager import StrategyManager

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
) -> tuple[str, str]:
    start_time = time.time()

    dataset_mgr = DatasetManager()
    strategy_mgr = StrategyManager()

    logger.info(
        f"Running {model_name} + {strategy_name} + {dataset_name} "
        f"(limit={limit}, top_k={top_k})"
    )

    # ===== Dataset Load (BUNDLE ONLY) =====
    bundle, chosen_year = dataset_mgr.load_bundle(
        name=dataset_name,
        cfg=dataset_cfg,
        limit=limit,
    )

    # Some strategies derive dataset_name from dataset_cfg["year"] for TREC DL.
    if dataset_name.startswith("trecdl") and chosen_year not in (None, "", "None"):
        dataset_cfg = dict(dataset_cfg)
        dataset_cfg["year"] = int(chosen_year)

    n_queries = len(bundle.queries_df) if getattr(bundle, "queries_df", None) is not None else 0
    n_docs = len(bundle.corpus_df) if getattr(bundle, "corpus_df", None) is not None else 0
    n_qrels = len(bundle.qrels_df) if getattr(bundle, "qrels_df", None) is not None else 0

    logger.info(
        f"Loaded {dataset_name} (year={chosen_year}) "
        f"with {n_queries} queries, {n_docs} docs, {n_qrels} qrels."
    )

    # ===== Run Strategy (BUNDLE ONLY) =====
    logger.info(f"Executing strategy: {strategy_name} ...")
    answers = strategy_mgr.run_strategy(
        name=strategy_name,
        bundle=bundle,
        model_cfg=model_cfg,
        strategy_cfg=strategy_cfg,
        dataset_cfg=dataset_cfg,
        top_k=top_k,
    )

    # ----- TOTAL EXPERIMENT TIME (attach to each JSONL row) -----
    elapsed = time.time() - start_time
    for a in answers:
        a["experiment_time_sec"] = round(elapsed, 2)

    # ===== Save Results =====
    persona_tag = ""
    if strategy_cfg.get("persona_key"):
        persona_tag = "_persona-" + str(strategy_cfg["persona_key"]).replace(" ", "").replace(",", "-")

    # If dataset_name already contains the year (trecdl_2019), don't append again
    if chosen_year not in (None, "", "None") and (str(chosen_year) not in str(dataset_name)):
        ds_tag = f"{dataset_name}{chosen_year}"
    else:
        ds_tag = dataset_name

    run_id = create_run_id(model_name, strategy_name, ds_tag)

    results_dir = OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / f"{run_id}{persona_tag}.jsonl"
    save_jsonl(answers, output_path)

    logger.info(f"Saved results → {output_path}")
    logger.info(f"Experiment completed in {elapsed:.2f}s ({len(answers)} results).")

    return str(output_path), chosen_year


# ---------------------------
# Main CLI Entrypoint
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Run IR experiments")
    parser.add_argument(
        "--models", type=str, required=True, default="none",
        help="Comma-separated list of model names"
    )
    parser.add_argument(
        "--strategies", type=str, required=True,
        help="Comma-separated list of strategy names"
    )
    parser.add_argument(
        "--datasets", type=str, required=True,
        help="Comma-separated list of dataset names"
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Limit number of queries (default=10)"
    )
    parser.add_argument(
        "--top_k", type=int, default=None,
        help="Top-k documents to retrieve (default=20 or from strategy)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Optional custom output dir"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without running"
    )
    parser.add_argument(
        "--persona", type=str, default=None,
        help="Comma-separated persona keys (used by persona/human-proxy strategies)"
    )

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
    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    strategy_names = [s.strip() for s in args.strategies.split(",") if s.strip()]

    # ----- Validation Helpers -----
    def get_model(name):
        if name == "none" or name is None:
            return {"name": "none", "model_id": None}
        for m in models_cfg:
            if m["name"] == name:
                return m
        raise ValueError(f"Model '{name}' not found in models.yaml")

    def get_strategy(name):
        if name in strategies_cfg:
            return dict(strategies_cfg[name])  # COPY to avoid mutation across runs
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

    if args.persona:
        for exp in experiments:
            if exp["strategy_name"] == "human_proxy_rag":
                exp["strategy_cfg"]["persona_key"] = args.persona

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
            result_path, chosen_year = run_single_experiment(
                model_cfg=exp["model_cfg"],
                strategy_cfg=exp["strategy_cfg"],
                dataset_cfg=exp["dataset_cfg"],
                dataset_name=exp["dataset_name"],
                strategy_name=exp["strategy_name"],
                model_name=exp["model_name"],
                limit=args.limit,
                top_k=args.top_k or exp["strategy_cfg"].get("top_k", 20),
            )

            exp_time = time.time() - exp_start
            tracker.add_run({
                "run_id": Path(result_path).stem,
                "dataset": exp["dataset_name"],
                "year": chosen_year,
                "strategy": exp["strategy_name"],
                "model": exp["model_name"],
                "persona_key": exp["strategy_cfg"].get("persona_key"),
                "limit": args.limit,
                "top_k": args.top_k or exp["strategy_cfg"].get("top_k", 20),
                "output_path": str(result_path),
                "duration_sec": round(exp_time, 2)
            })
            logger.info(
                f"Completed {exp['model_name']} + {exp['strategy_name']} "
                f"in {exp_time:.2f}s\n"
            )
        except Exception as e:
            logger.error(f"Experiment {i} failed: {e}")
            continue

    total_time = time.time() - total_start
    logger.info(f"All experiments done in {total_time:.2f}s.")
    logger.info(f"Metadata saved → {OUTPUT_DIR / 'experiment_metadata.json'}")


if __name__ == "__main__":
    main()
