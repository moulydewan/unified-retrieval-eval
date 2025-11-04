import logging
from pathlib import Path
from src.datasets.trecdl import TRECDLAdapter
from src.experiments.rag import retrieve_bm25, rag_synthesis
from src.utils import setup_logging, save_jsonl, create_run_id, ExperimentTracker

# Setup experiment logging
logger = setup_logging(Path("outputs/logs"), "rag_test")
tracker = ExperimentTracker(Path("outputs"))

if __name__ == "__main__":
    logger.info("Starting RAG experiment (BM25 + Claude 4.5 Haiku)")

    # Load dataset
    trec = TRECDLAdapter(year=2019, mode="passage")
    data = trec.load(limit=1)
    df = trec.trec_df(data)
    logger.info(f"Loaded {len(df)} rows from TREC DL 2019")

    # Run BM25 retrieval
    bm25_df = retrieve_bm25(df, top_k=5)
    logger.info(f"BM25 retrieved {len(bm25_df)} rows")

    # Run RAG synthesis
    model_arn = (
        "arn:aws:bedrock:us-east-1:033792130535:inference-profile/"
        "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )
    answers = rag_synthesis(
        bm25_df=bm25_df,
        model_id=model_arn,
        temperature=0.7,
        top_k=5,
        prompt_path=Path("prompts/rag_prompt.txt"),
    )
    logger.info(f"Generated {len(answers)} answers")

    # Save results
    run_id = create_run_id("claude-4-5-haiku", "rag", "trecdl")
    output_path = Path("outputs/results") / f"{run_id}.jsonl"
    save_jsonl(answers, output_path)
    logger.info(f"Results saved to: {output_path}")

    # Track metadata
    tracker.add_run({
        "run_id": run_id,
        "dataset": "trecdl",
        "strategy": "rag",
        "model": "claude-4-5-haiku",
        "temperature": 0.7,
        "top_k": 5,
        "num_queries": len(answers),
        "output_path": str(output_path),
    })
    logger.info("Experiment metadata logged successfully")
