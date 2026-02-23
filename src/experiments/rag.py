# src/experiments/rag.py
import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from jinja2 import Template

from src.generation_backend.aws_bedrock_backend import generate_bedrock_model
from src.datasets.datasetbundle import IRDatasetBundle
from src.experiments.pyserini_bm25 import retrieve_bm25_pyserini  

logger = logging.getLogger(__name__)


# ---------------------------
# Helpers
# ---------------------------
def truncate_text(s: str, max_chars: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= max_chars else s[:max_chars].rstrip() + "…"


def load_prompt_template(prompt_path: Path) -> Template:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
    return Template(prompt_path.read_text(encoding="utf-8"))


# ---------------------------
# RAG synthesis
# ---------------------------
def rag_synthesis(
    bm25_df: pd.DataFrame,
    *,
    model_id: str,
    temperature: float,
    max_tokens: int,
    gen_top_k: int,
    prompt_path: Path,
    max_passage_chars: int,
) -> List[Dict[str, Any]]:
    req = {"qid", "query", "docid", "passage", "rank", "bm25_score"}
    assert req.issubset(bm25_df.columns), bm25_df.columns
    assert isinstance(gen_top_k, int) and gen_top_k > 0

    template = load_prompt_template(prompt_path)

    cols = ["qid", "query"]
    if "complexity" in bm25_df.columns:
        cols.append("complexity")
    unique_queries = bm25_df[cols].drop_duplicates()
    logger.info(f"RAG synthesis: {len(unique_queries)} queries, model={model_id}, gen_top_k={gen_top_k}")

    outputs: List[Dict[str, Any]] = []
    for _, qrow in unique_queries.iterrows():
        qid, query = qrow["qid"], qrow["query"]
        complexity = qrow["complexity"] if "complexity" in unique_queries.columns else None

        # ---- always keep top-10 for evaluation ----
        ranked = bm25_df[bm25_df["qid"] == qid].sort_values("rank")
        subset_eval = ranked.head(10)              # <- saved in JSONL for eval
        subset_prompt = subset_eval.head(gen_top_k)  # <- used only for prompt

        # Prompt docs (truncate for token budget)
        docs_for_prompt: List[Dict[str, Any]] = []
        for _, d in subset_prompt.iterrows():
            docs_for_prompt.append(
                {
                    "docid": str(d["docid"]),
                    "rank": int(d["rank"]),
                    "score": float(d["bm25_score"]),
                    "passage": truncate_text(d["passage"], max_passage_chars),
                }
            )

        # Eval docs (top-10; keep passage as-is so eval isn't affected by truncation)
        docs_for_eval: List[Dict[str, Any]] = []
        for _, d in subset_eval.iterrows():
            docs_for_eval.append(
                {
                    "docid": str(d["docid"]),
                    "rank": int(d["rank"]),
                    "score": float(d["bm25_score"]),
                    "passage": str(d["passage"]),
                }
            )

        prompt = template.render(query=query, docs=docs_for_prompt, top_k=gen_top_k)

        error = None
        usage = None
        try:
            gen = generate_bedrock_model(
                model_id=model_id,
                prompt=prompt,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                return_meta=True,  # <-- NEW
            )
            answer_text = (gen.get("text") or "").strip()
            usage = gen.get("usage")  # <-- NEW (may be None)
        except Exception as e:
            logger.error(f"Generation failed for qid={qid}: {e}")
            answer_text = ""
            error = f"{type(e).__name__}: {e}"

        outputs.append(
            {
                "qid": qid,
                "query": query,
                "complexity": complexity,
                "answer": answer_text,
                "retrieved_docs": docs_for_eval,
                "usage": usage,
                "prompt_chars": len(prompt),
                "answer_chars": len(answer_text),
                "error": error,
            }
        )

    return outputs


# ---------------------------
# Shared BM25 entrypoint (for ALL strategies)
# ---------------------------
def retrieve_bm25(
    bundle: IRDatasetBundle,
    dataset_cfg: Dict[str, Any],
    strategy_cfg: Dict[str, Any],
    *,
    top_k: int,
) -> pd.DataFrame:
    """
    Shared BM25 retrieval.

    Index selection:
      - dataset_cfg provides defaults
      - strategy_cfg can override

    Requires ONE of:
      - pyserini_index_dir
      - pyserini_prebuilt_index
    """
    assert {"qid", "query"}.issubset(bundle.queries_df.columns), bundle.queries_df.columns

    index_dir = strategy_cfg.get("pyserini_index_dir") or dataset_cfg.get("pyserini_index_dir")
    prebuilt = strategy_cfg.get("pyserini_prebuilt_index") or dataset_cfg.get("pyserini_prebuilt_index")

    if not index_dir and not prebuilt:
        raise ValueError(
            "BM25 requires pyserini_index_dir or pyserini_prebuilt_index "
            "(set in datasets.yaml)."
        )

    return retrieve_bm25_pyserini(
        bundle.queries_df,
        top_k=int(top_k),
        index_dir=index_dir,
        prebuilt_index=prebuilt,
        k1=float(strategy_cfg.get("bm25_k1", 0.9)),
        b=float(strategy_cfg.get("bm25_b", 0.4)),
    )


# ---------------------------
# Strategy runner: BM25 -> RAG
# ---------------------------
def run_bm25_rag(
    bundle: IRDatasetBundle,
    model_cfg: Dict[str, Any],
    strategy_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    *,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Strategy: rag baseline (BM25 via Pyserini + single-shot generation)
    """
    assert {"qid", "query"}.issubset(bundle.queries_df.columns), bundle.queries_df.columns

    model_id = model_cfg.get("model_id")
    if not model_id:
        raise ValueError("rag requires model_cfg['model_id'].")

    retrieval_top_k = int(strategy_cfg.get("retrieval_top_k", top_k))
    gen_top_k = int(strategy_cfg.get("gen_top_k", top_k))

    bm25_df = retrieve_bm25(
        bundle=bundle,
        dataset_cfg=dataset_cfg,
        strategy_cfg=strategy_cfg,
        top_k=retrieval_top_k,
    )

    if "complexity" in bundle.queries_df.columns:
        bm25_df = bm25_df.merge(
            bundle.queries_df[["qid", "complexity"]],
            on="qid",
            how="left",
        )

    return rag_synthesis(
        bm25_df=bm25_df,
        model_id=model_id,
        temperature=float(strategy_cfg.get("temperature", 0.7)),
        max_tokens=int(model_cfg.get("max_tokens", 512)),
        gen_top_k=gen_top_k,
        prompt_path=Path(strategy_cfg.get("prompt_path", "prompts/rag_prompt.txt")),
        max_passage_chars=int(strategy_cfg.get("max_passage_chars", 1200)),
    )
