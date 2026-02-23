# src/experiments/human_proxy_rag.py
import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from jinja2 import Template

from src.generation_backend.aws_bedrock_backend import generate_bedrock_model
from src.datasets.datasetbundle import IRDatasetBundle
from src.experiments.rag import retrieve_bm25, truncate_text  # reuse shared BM25 + helper

logger = logging.getLogger(__name__)


def load_prompt_template(prompt_path: Path) -> Template:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
    return Template(prompt_path.read_text(encoding="utf-8"))


def persona_rag_synthesis(
    bm25_df: pd.DataFrame,
    *,
    model_id: str,
    temperature: float,
    max_tokens: int,
    gen_top_k: int,
    prompt_path: Path,
    max_passage_chars: int,
    persona: Dict[str, Any],
) -> List[Dict[str, Any]]:
    req = {"qid", "query", "docid", "passage", "rank", "bm25_score"}
    assert req.issubset(bm25_df.columns), bm25_df.columns
    assert isinstance(gen_top_k, int) and gen_top_k > 0

    required_persona = {"persona_name", "persona_description", "persona_roles"}
    missing = sorted(required_persona - set(persona.keys()))
    if missing:
        raise ValueError(f"Persona missing required fields: {missing}")

    template = load_prompt_template(prompt_path)

    cols = ["qid", "query"]
    if "complexity" in bm25_df.columns:
        cols.append("complexity")
    unique_queries = bm25_df[cols].drop_duplicates()

    outputs: List[Dict[str, Any]] = []
    for _, qrow in unique_queries.iterrows():
        qid, query = qrow["qid"], qrow["query"]
        complexity = qrow["complexity"] if "complexity" in unique_queries.columns else None

        subset = bm25_df[bm25_df["qid"] == qid].sort_values("rank").head(gen_top_k)

        docs_for_prompt: List[Dict[str, Any]] = []
        for _, d in subset.iterrows():
            docs_for_prompt.append(
                {
                    "docid": str(d["docid"]),
                    "rank": int(d["rank"]),
                    "score": float(d["bm25_score"]),
                    "passage": truncate_text(d["passage"], max_passage_chars),
                }
            )

        prompt = template.render(
            query=query,
            docs=docs_for_prompt,
            top_k=gen_top_k,
            persona_name=persona["persona_name"],
            persona_description=persona["persona_description"],
            persona_roles=persona["persona_roles"],
        )

        error = None
        usage = None
        try:
            gen = generate_bedrock_model(
                model_id=model_id,
                prompt=prompt,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                return_meta=True,  # <-- add
            )
            answer_text = (gen.get("text") or "").strip()
            usage = gen.get("usage")  # <-- add (may be None)
        except Exception as e:
            logger.error(f"Generation failed for qid={qid} persona={persona.get('persona_name')}: {e}")
            answer_text = ""
            error = f"{type(e).__name__}: {e}"

        outputs.append(
            {
                "qid": qid,
                "query": query,
                "complexity": complexity,
                "persona": persona["persona_name"],
                "answer": answer_text,
                "retrieved_docs": docs_for_prompt,
                "usage": usage,                 # <-- add
                "prompt_chars": len(prompt),    # <-- optional but matches rag.py
                "answer_chars": len(answer_text),# <-- optional but matches rag.py
                "error": error,
            }
        )

    return outputs


def run_human_proxy_rag(
    bundle: IRDatasetBundle,
    model_cfg: Dict[str, Any],
    strategy_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    *,
    top_k: int,
    persona: Dict[str, Any] | List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """
    Strategy 3: Human-proxy with personas
      - BM25 via Pyserini ONCE (shared retrieve_bm25)
      - generation per persona
    """
    if persona is None:
        raise ValueError("Persona must be provided by StrategyManager")

    model_id = model_cfg.get("model_id")
    if not model_id:
        raise ValueError("human_proxy_rag requires model_cfg['model_id'] (do not use --models none).")

    personas = persona if isinstance(persona, list) else [persona]

    retrieval_top_k = int(strategy_cfg.get("retrieval_top_k", top_k))
    gen_top_k = int(strategy_cfg.get("gen_top_k", top_k))

    # ---- BM25 once (Pyserini, shared) ----
    bm25_df = retrieve_bm25(
        bundle=bundle,
        dataset_cfg=dataset_cfg,
        strategy_cfg=strategy_cfg,
        top_k=retrieval_top_k,
    )

    # attach complexity if present (same as rag.py)
    if "complexity" in bundle.queries_df.columns:
        bm25_df = bm25_df.merge(
            bundle.queries_df[["qid", "complexity"]],
            on="qid",
            how="left",
        )

    prompt_path = Path(strategy_cfg.get("prompt_path", "prompts/persona_prompt.txt"))
    max_passage_chars = int(strategy_cfg.get("max_passage_chars", 1200))

    # ---- synth per persona ----
    outputs: List[Dict[str, Any]] = []
    for p in personas:
        outputs.extend(
            persona_rag_synthesis(
                bm25_df=bm25_df,
                model_id=model_id,
                temperature=float(strategy_cfg.get("temperature", 0.7)),
                max_tokens=int(model_cfg.get("max_tokens", 512)),
                gen_top_k=gen_top_k,
                prompt_path=prompt_path,
                max_passage_chars=max_passage_chars,
                persona=p,
            )
        )

    return outputs
