# src/experiments/agent_hierarchy.py
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.agents.orchestrator_agent import OrchestratorAgent
from src.agents.query_reformulation_agent import QueryReformulationAgent
from src.agents.dense_retrieval_agent import DenseRetrieverAgent
from src.agents.fact_verification_agent import FactVerificationAgent

logger = logging.getLogger(__name__)


def _safe_dataset_name(name: str) -> str:
    return str(name).replace("/", "_").replace(":", "_")


# -------------------------
# Token helpers (same style as peer/functional)
# -------------------------
def _extract_payload_and_usage(out):
    if isinstance(out, dict):
        usage = out.get("usage")
        if "subqueries" in out:
            return out.get("subqueries"), usage
        if "verified_facts" in out:
            return out.get("verified_facts"), usage

        payload = out.get("text")
        # NEW: if text is a list, treat it as subqueries payload
        if isinstance(payload, list):
            return payload, usage
        return payload, usage
    return out, None



def _sum_usage(usages: List[Dict[str, Any]]) -> Dict[str, int]:
    def get(u, *keys):
        for k in keys:
            if k in (u or {}):
                return int(u[k] or 0)
        return 0

    input_tokens = sum(get(u, "inputTokens", "input_tokens") for u in usages)
    output_tokens = sum(get(u, "outputTokens", "output_tokens") for u in usages)

    return {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "totalTokens": input_tokens + output_tokens,
    }


def run_agent_hierarchy_strategy(
    bundle,
    model_cfg,
    strategy_cfg,
    dataset_cfg,
    top_k=5,
):
    if bundle is None:
        raise ValueError("Strategy 4 requires an IRDatasetBundle (bundle).")

    if not {"qid", "query"}.issubset(bundle.queries_df.columns):
        raise ValueError("bundle.queries_df must include ['qid','query'].")

    if not {"docid", "passage"}.issubset(bundle.corpus_df.columns):
        raise ValueError("bundle.corpus_df must include ['docid','passage'].")

    model_cfg = model_cfg or {}
    strategy_cfg = strategy_cfg or {}
    dataset_cfg = dataset_cfg or {}

    top_k_final = int(strategy_cfg.get("top_k", top_k))

    if "year" in dataset_cfg:
        dataset_name = f"trecdl_{dataset_cfg['year']}"
    elif dataset_cfg.get("pyserini_prebuilt_index"):
        dataset_name = dataset_cfg["pyserini_prebuilt_index"]
    else:
        raise ValueError(
            "dataset_cfg must include 'year' (trecdl) or "
            "'pyserini_prebuilt_index' (beir/nq)."
        )

    # ---- Strategy 4 uses judged FAISS only ----
    safe_name = _safe_dataset_name(dataset_name)
    index_path = Path("outputs") / "dense" / f"{safe_name}_judged" / "faiss_index.bin"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}. Run Strategy 2 (dense retrieval) once to build caches."
        )

    logger.info(
        f"[hierarchy_agents] Loaded bundle: {len(bundle.queries_df)} queries, {len(bundle.corpus_df)} docs "
        f"(faiss={index_path})"
    )

    return run_agent_hierarchy(
        queries_df=bundle.queries_df,
        corpus_df=bundle.corpus_df,
        top_k=top_k_final,
        model_cfg=model_cfg,
        strategy_cfg=strategy_cfg,
        index_path=str(index_path),
        dataset_name=dataset_name,
    )


def run_agent_hierarchy(
    queries_df,
    corpus_df,
    top_k=5,
    model_cfg=None,
    strategy_cfg=None,
    index_path=None,
    dataset_name="dataset",
):
    if not index_path:
        raise ValueError("index_path is required. Run via run_agent_hierarchy_strategy().")

    model_cfg = model_cfg or {}
    strategy_cfg = strategy_cfg or {}

    bedrock_model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

    orchestrator = OrchestratorAgent(
        model_id=bedrock_model_id,
        temperature=model_cfg.get("temperature", 0.7),
        max_tokens=model_cfg.get("max_tokens", 4096),
    )

    reformulator = QueryReformulationAgent(
        model_id=bedrock_model_id,
        temperature=model_cfg.get("temperature", 0.7),
        max_tokens=min(int(model_cfg.get("max_tokens", 512)), 1024),
    )

    embedding_model = model_cfg.get("retriever", "BAAI/bge-large-en-v1.5")

    # Dense retrieval settings
    top_k_dense = int(strategy_cfg.get("top_k_dense", 50))
    reranker_batch_size = int(strategy_cfg.get("reranker_batch_size", 16))

    retriever = DenseRetrieverAgent(
        df=corpus_df,
        doc_col="passage",
        docid_col="docid",
        model_name=embedding_model,
        index_path=index_path,
        reranker_batch_size=reranker_batch_size,
    )

    verifier = FactVerificationAgent(
        model_id=bedrock_model_id,
        temperature=model_cfg.get("temperature", 0.2),
        max_tokens=min(int(model_cfg.get("max_tokens", 1024)), 2048),
    )

    results = []

    cols = ["qid", "query"]
    if "complexity" in queries_df.columns:
        cols.append("complexity")

    unique_queries = queries_df[cols].drop_duplicates()

    for _, row in unique_queries.iterrows():
        qid = str(row["qid"])
        query = row["query"]
        complexity = row["complexity"] if "complexity" in unique_queries.columns else None

        usage_calls: List[Dict[str, Any]] = []

        # 1) Plan
        plan_out = orchestrator.plan(query)
        plan, u = _extract_payload_and_usage(plan_out)
        if isinstance(u, dict):
            usage_calls.append(u)

        # 2) Rewrite -> subqueries (CAPTURE USAGE)
        subq_out = reformulator.rewrite_with_usage(query)
        subqueries = subq_out.get("text", [query, query, query])
        u = subq_out.get("usage")
        if isinstance(u, dict):
            usage_calls.append(u)

        # Normalize subqueries if returned as a string (fallback)
        if isinstance(subqueries, str):
            subqueries = [s.strip() for s in subqueries.splitlines() if s.strip()]
        if not isinstance(subqueries, list):
            subqueries = [str(subqueries)] if subqueries else []

        retrieved_docs = []
        seen = set()

        for sq in subqueries:
            docs = retriever.retrieve(query=sq, k=top_k, top_k_dense=top_k_dense)
            for d in docs:
                docid = str(d["docid"])
                if docid in seen:
                    continue
                seen.add(docid)
                retrieved_docs.append(
                    {
                        "docid": docid,
                        "passage": d.get("passage", ""),
                        "rank": len(retrieved_docs) + 1,  # temporary
                        "source_subquery": sq,
                    }
                )

        # 🔽 ADD THIS HERE (after the loop over subqueries ends)

        retrieved_docs = retrieved_docs[:top_k]
        for i, d in enumerate(retrieved_docs, start=1):
            d["rank"] = i


        # 3) Verify facts
        ver_out = verifier.verify(query, retrieved_docs)
        verified_facts, u = _extract_payload_and_usage(ver_out)
        if isinstance(u, dict):
            usage_calls.append(u)

        # 4) Synthesize final answer
        synth_out = orchestrator.synthesize(query, verified_facts)
        final_answer, u = _extract_payload_and_usage(synth_out)
        if isinstance(u, dict):
            usage_calls.append(u)

        usage_total = _sum_usage(usage_calls)

        results.append(
            {
                "qid": qid,
                "query": query,
                **({"complexity": complexity} if complexity is not None else {}),
                "plan": plan,
                "subqueries": subqueries,
                "retrieved_docs": retrieved_docs,
                "verified_facts": verified_facts,
                "final_answer": final_answer,
                # ---- tokens (same pattern as peer/functional) ----
                "usage_calls": usage_calls,
                "usage_total": usage_total,
                "usage": usage_total,  # alias for eval/tokenizer convenience
                "strategy": "hierarchy_agents",
                "metadata": {
                    "dataset": dataset_name,
                    "faiss_index_path": index_path,
                    "embedding_model": embedding_model,
                    "top_k_dense": top_k_dense,
                    "reranker_batch_size": reranker_batch_size,
                },
            }
        )
    return results
