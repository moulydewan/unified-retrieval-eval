
# src/experiments/agent_peer.py
import logging
from pathlib import Path

from src.agents.dense_retrieval_agent import DenseRetrieverAgent
from src.agents.exploratory_agent import Exploratory_Agent
from src.agents.analytical_agent import Analytical_Agent
from src.agents.verification_agent import Verification_Agent
from src.agents.judge_agent import Judge_Agent

logger = logging.getLogger(__name__)

def _extract_text_and_usage(out):
    """
    Supports:
      - str
      - dict with {text, usage}
    """
    if isinstance(out, dict):
        return out.get("text", ""), out.get("usage")
    return out, None


def _sum_usage(usages):
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



def _safe_dataset_name(name: str) -> str:
    return str(name).replace("/", "_").replace(":", "_")


def run_peer_agents(
    bundle,
    model_cfg,
    strategy_cfg,
    dataset_cfg,
    top_k=5,
):
    """
    Strategy 5 – Peer-to-Peer Agents:
    Shared Dense Retrieval (judged FAISS) → A/B/C agents → Judge → Final Answer

    Mirrors agent_hierarchy.py:
      - requires IRDatasetBundle
      - determines dataset_name from dataset_cfg (trecdl year OR pyserini_prebuilt_index)
      - requires FAISS index built by Strategy 2
      - returns a list[dict] with qid/query/(optional complexity)/retrieved_docs/... + strategy + metadata
    """
    if bundle is None:
        raise ValueError("Strategy 5 requires an IRDatasetBundle (bundle).")

    if not {"qid", "query"}.issubset(bundle.queries_df.columns):
        raise ValueError("bundle.queries_df must include ['qid','query'].")

    if not {"docid", "passage"}.issubset(bundle.corpus_df.columns):
        raise ValueError("bundle.corpus_df must include ['docid','passage'].")

    model_cfg = model_cfg or {}
    strategy_cfg = strategy_cfg or {}
    dataset_cfg = dataset_cfg or {}

    top_k_final = int(strategy_cfg.get("top_k", top_k))

    # Match hierarchy_agents dataset_name resolution
    if "year" in dataset_cfg:
        dataset_name = f"trecdl_{dataset_cfg['year']}"
    elif dataset_cfg.get("pyserini_prebuilt_index"):
        dataset_name = dataset_cfg["pyserini_prebuilt_index"]
    else:
        raise ValueError(
            "dataset_cfg must include 'year' (trecdl) or "
            "'pyserini_prebuilt_index' (beir/nq)."
        )

    # Strategy 5 uses judged FAISS only (same as Strategy 4)
    safe_name = _safe_dataset_name(dataset_name)
    index_path = Path("outputs") / "dense" / f"{safe_name}_judged" / "faiss_index.bin"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}. Run Strategy 2 (dense retrieval) once to build caches."
        )

    logger.info(
        f"[peer_agents] Loaded bundle: {len(bundle.queries_df)} queries, {len(bundle.corpus_df)} docs "
        f"(faiss={index_path})"
    )

    return run_peer_agents_core(
        queries_df=bundle.queries_df,
        corpus_df=bundle.corpus_df,
        top_k=top_k_final,
        model_cfg=model_cfg,
        strategy_cfg=strategy_cfg,
        index_path=str(index_path),
        dataset_name=dataset_name,
    )


def run_peer_agents_core(
    queries_df,
    corpus_df,
    top_k=5,
    model_cfg=None,
    strategy_cfg=None,
    index_path=None,
    dataset_name="dataset",
):
    if not index_path:
        raise ValueError("index_path is required. Run via run_peer_agents().")

    model_cfg = model_cfg or {}
    strategy_cfg = strategy_cfg or {}

    embedding_model = model_cfg.get("retriever", "BAAI/bge-large-en-v1.5")

    # NOTE: DenseRetrieverAgent loads the embedding model once here (good)
    retriever = DenseRetrieverAgent(
        df=corpus_df,
        doc_col="passage",
        docid_col="docid",
        model_name=embedding_model,
        index_path=index_path,
    )

    exploratory_agent = Exploratory_Agent()
    analytical_agent = Analytical_Agent()
    verification_agent = Verification_Agent()
    judge_agent = Judge_Agent()

    results = []

    cols = ["qid", "query"]
    if "complexity" in queries_df.columns:
        cols.append("complexity")
    unique_queries = queries_df[cols].drop_duplicates()

    for _, row in unique_queries.iterrows():
        qid = str(row["qid"])
        query = row["query"]
        complexity = row["complexity"] if "complexity" in unique_queries.columns else None

        # Shared retrieval once per query
        docs = retriever.retrieve(query=query, k=top_k)

        # Convert retriever output -> framework retrieved_docs
        # keep score, add rank, ensure docid is str, dedupe
        retrieved_docs = []
        seen = set()
        for d in docs:
            docid = str(d["docid"])
            if docid in seen:
                continue
            seen.add(docid)
            retrieved_docs.append(
                {
                    "docid": docid,
                    "passage": d.get("passage", ""),
                    "rank": len(retrieved_docs) + 1,
                    "score": float(d.get("score", 0.0)),                 # final score (cross)
                    "cross_score": float(d.get("cross_score", d.get("score", 0.0))),
                    "dense_score": float(d.get("dense_score", 0.0)),
                    "dense_rank": int(d.get("dense_rank", len(retrieved_docs) + 1)),
                    "rerank": int(d.get("rerank", len(retrieved_docs) + 1)),
                }
            )


        usage_calls = []

        # Exploratory Agent
        exp_out = exploratory_agent.run(query, retrieved_docs)
        exploratory_output, u = _extract_text_and_usage(exp_out)
        if u:
            usage_calls.append(u)

        # Analytical Agent
        ana_out = analytical_agent.run(query, retrieved_docs)
        analytical_output, u = _extract_text_and_usage(ana_out)
        if u:
            usage_calls.append(u)

        # Verification Agent
        ver_out = verification_agent.run(query, retrieved_docs)
        verification_output, u = _extract_text_and_usage(ver_out)
        if u:
            usage_calls.append(u)

        # Judge Agent
        judge_out = judge_agent.evaluate(
            query=query,
            exploratory_output=exploratory_output,
            analytical_output=analytical_output,
            verification_output=verification_output,
        )
        final_answer, u = _extract_text_and_usage(judge_out)
        if u:
            usage_calls.append(u)

        usage_total = _sum_usage(usage_calls)


        results.append(
            {
                "qid": qid,
                "query": query,
                **({"complexity": complexity} if complexity is not None else {}),
                "retrieved_docs": retrieved_docs,
                "exploratory_answer": exploratory_output,
                "analytical_answer": analytical_output,
                "verification_answer": verification_output,
                "final_answer": final_answer,
                "usage": usage_total, 
                "usage_calls": usage_calls,     
                "usage_total": usage_total,     
                "strategy": "peer_agents",
                "metadata": {
                    "dataset": dataset_name,
                    "faiss_index_path": index_path,
                    "embedding_model": embedding_model,
                    "top_k": int(top_k),
                },
            }
        )

    return results