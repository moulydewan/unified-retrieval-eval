# # src/agents/agent_functional.py

# import logging
# from pathlib import Path
# from typing import Dict, Any, List

# import numpy as np
# import pandas as pd
# from FlagEmbedding import FlagReranker

# from src.agents.dense_retrieval_agent import DenseRetrieverAgent
# from src.agents.query_reformulation_agent import QueryReformulationAgent
# from src.agents.analyzer_agent import AnalyzerAgent
# from src.agents.synthesizer_agent import SynthesizerAgent
# from src.agents.validator_agent import ValidatorAgent
# from src.experiments.pyserini_bm25 import retrieve_bm25_pyserini

# logger = logging.getLogger(__name__)


# def _extract_text_and_usage(out):
#     if isinstance(out, dict):
#         return out.get("text", ""), out.get("usage")
#     return out, None


# def _sum_usage(usages):
#     def get(u, *keys):
#         for k in keys:
#             if k in (u or {}):
#                 return int(u[k] or 0)
#         return 0

#     input_tokens = sum(get(u, "inputTokens", "input_tokens") for u in usages)
#     output_tokens = sum(get(u, "outputTokens", "output_tokens") for u in usages)

#     return {
#         "inputTokens": input_tokens,
#         "outputTokens": output_tokens,
#         "totalTokens": input_tokens + output_tokens,
#     }


# def _safe_dataset_name(name: str) -> str:
#     return str(name).replace("/", "_").replace(":", "_")


# def run_functional_agents(
#     bundle,
#     model_cfg: Dict[str, Any],
#     strategy_cfg: Dict[str, Any],
#     dataset_cfg: Dict[str, Any],
#     top_k: int = 10,
# ):
#     if bundle is None:
#         raise ValueError("Strategy 6 requires an IRDatasetBundle.")

#     model_cfg = model_cfg or {}
#     strategy_cfg = strategy_cfg or {}
#     dataset_cfg = dataset_cfg or {}

#     bm25_top_k = int(strategy_cfg.get("bm25_top_k", 10))
#     dense_top_k = int(strategy_cfg.get("dense_top_k", 10))
#     reformulate_k = int(strategy_cfg.get("reformulate_k", 3))
#     reform_dense_top_k = int(strategy_cfg.get("reform_dense_top_k", 10))
#     gather_top_k_total = int(strategy_cfg.get("gather_top_k_total", 10))

#     # NEW: batching for global rerank
#     reranker_batch_size = int(strategy_cfg.get("reranker_batch_size", 16))

#     # Dataset name resolution
#     if "year" in dataset_cfg:
#         dataset_name = f"trecdl_{dataset_cfg['year']}"
#     elif dataset_cfg.get("pyserini_prebuilt_index"):
#         dataset_name = dataset_cfg["pyserini_prebuilt_index"]
#     else:
#         raise ValueError("dataset_cfg must include 'year' or 'pyserini_prebuilt_index'.")

#     safe_name = _safe_dataset_name(dataset_name)
#     faiss_path = Path("outputs") / "dense" / f"{safe_name}_judged" / "faiss_index.bin"
#     if not faiss_path.exists():
#         raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

#     corpus_df = bundle.corpus_df.copy()
#     corpus_df["docid"] = corpus_df["docid"].astype(str)
#     docid_to_passage = dict(zip(corpus_df["docid"], corpus_df["passage"].astype(str)))

#     embedding_model = model_cfg.get("retriever", "BAAI/bge-large-en-v1.5")
#     reranker_name = model_cfg.get("reranker", "BAAI/bge-reranker-large")

#     dense_retriever = DenseRetrieverAgent(
#         df=corpus_df,
#         doc_col="passage",
#         docid_col="docid",
#         model_name=embedding_model,
#         index_path=str(faiss_path),
#     )

#     reformulator = QueryReformulationAgent(
#         temperature=float(strategy_cfg.get("reformulator_temperature", 0.2)),
#         max_tokens=int(strategy_cfg.get("reformulator_max_tokens", 256)),
#     )

#     reranker = FlagReranker(reranker_name, use_fp16=True)

#     analyzer = AnalyzerAgent()
#     synthesizer = SynthesizerAgent()
#     validator = ValidatorAgent()

#     cols = ["qid", "query"]
#     if "complexity" in bundle.queries_df.columns:
#         cols.append("complexity")

#     queries_df = bundle.queries_df[cols].drop_duplicates().copy()
#     queries_df["qid"] = queries_df["qid"].astype(str)

#     results: List[Dict[str, Any]] = []

#     for _, row in queries_df.iterrows():
#         qid = row["qid"]
#         query = row["query"]
#         complexity = row.get("complexity")

#         usage_calls = []

#         # -------------------------
#         # Gatherer (BM25 + Dense)
#         # -------------------------
#         one_q_df = pd.DataFrame([{"qid": qid, "query": query}])

#         bm25_df = retrieve_bm25_pyserini(
#             one_q_df,
#             top_k=bm25_top_k,
#             index_dir=dataset_cfg.get("pyserini_index_dir"),
#             prebuilt_index=dataset_cfg.get("pyserini_prebuilt_index"),
#         )

#         bm25_docs = []
#         for _, hit in bm25_df.iterrows():
#             docid = str(hit.get("docid", "")).strip()
#             if not docid:
#                 continue
#             passage = hit.get("passage") or docid_to_passage.get(docid, "")
#             if not passage:
#                 continue
#             bm25_docs.append(
#                 {
#                     "docid": docid,
#                     "passage": passage,
#                     "source": "bm25",
#                     "bm25_score": float(hit.get("bm25_score", 0.0)),
#                 }
#             )

#         dense_hits = dense_retriever.retrieve(query=query, k=dense_top_k)
#         dense_docs = [
#             {
#                 "docid": str(d["docid"]),
#                 "passage": d.get("passage", ""),
#                 "source": "dense",
#                 "dense_score": float(d.get("score", 0.0)),
#             }
#             for d in dense_hits
#         ]

#         # -------------------------
#         # Reformulation
#         # -------------------------
#         ref_out = reformulator.rewrite_with_usage(query)
#         subqueries, u = _extract_text_and_usage(ref_out)

#         subqueries = (subqueries or [])[:reformulate_k]
#         if u:
#             usage_calls.append(u)

#         reform_dense_docs = []
#         for sq in subqueries:
#             for d in dense_retriever.retrieve(query=sq, k=reform_dense_top_k):
#                 reform_dense_docs.append(
#                     {
#                         "docid": str(d["docid"]),
#                         "passage": d.get("passage", ""),
#                         "source": "reform_dense",
#                         "subquery": sq,
#                         "dense_score": float(d.get("score", 0.0)),
#                     }
#                 )

#         # Merge + dedupe (docid-level)
#         merged = {d["docid"]: d for d in (bm25_docs + dense_docs + reform_dense_docs)}
#         merged_docs = list(merged.values())

#         # -------------------------
#         # Global rerank (BATCHED)
#         # -------------------------
#         pairs = [[query, d["passage"]] for d in merged_docs]
#         cross_scores: List[float] = []

#         bs = max(1, int(reranker_batch_size))
#         for start in range(0, len(pairs), bs):
#             batch = pairs[start : start + bs]
#             out = reranker.compute_score(batch)
#             if isinstance(out, (list, tuple, np.ndarray)):
#                 cross_scores.extend([float(x) for x in out])
#             else:
#                 cross_scores.append(float(out))

#         if len(cross_scores) != len(merged_docs):
#             raise RuntimeError(
#                 f"[functional_agents] Reranker returned {len(cross_scores)} scores for {len(merged_docs)} docs."
#             )

#         scored = []
#         for d, s in zip(merged_docs, cross_scores):
#             d2 = dict(d)
#             d2["cross_score"] = float(s)
#             scored.append(d2)

#         scored.sort(key=lambda x: x["cross_score"], reverse=True)
#         retrieved_docs = scored[:gather_top_k_total]

#         # -------------------------
#         # Functional agents (collect usage)
#         # -------------------------
#         ana_out = analyzer.run_with_usage(query, retrieved_docs)
#         analysis_text, u = _extract_text_and_usage(ana_out)
#         if u:
#             usage_calls.append(u)

#         syn_out = synthesizer.run_with_usage(query, retrieved_docs, analysis_text)
#         draft, u = _extract_text_and_usage(syn_out)
#         if u:
#             usage_calls.append(u)

#         val_out = validator.run_with_usage(query, retrieved_docs, draft)
#         final_answer, u = _extract_text_and_usage(val_out)
#         if u:
#             usage_calls.append(u)

#         usage_total = _sum_usage(usage_calls)

#         results.append(
#             {
#                 "qid": qid,
#                 "query": query,
#                 **({"complexity": complexity} if complexity is not None else {}),
#                 "subqueries": subqueries,
#                 "retrieved_docs": retrieved_docs,
#                 "analysis": analysis_text,
#                 "final_answer": final_answer,
#                 "usage": usage_total,
#                 "usage_calls": usage_calls,
#                 "usage_total": usage_total,
#                 "strategy": "functional_agents",
#                 "metadata": {
#                     "dataset": dataset_name,
#                     "faiss_index_path": str(faiss_path),
#                     "embedding_model": embedding_model,
#                     "reranker_model": reranker_name,
#                     "bm25_top_k": bm25_top_k,
#                     "dense_top_k": dense_top_k,
#                     "reformulate_k": reformulate_k,
#                     "reform_dense_top_k": reform_dense_top_k,
#                     "gather_top_k_total": gather_top_k_total,
#                     "reranker_batch_size": reranker_batch_size,
#                 },
#             }
#         )

#     return results


# src/agents/agent_functional.py

import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from FlagEmbedding import FlagReranker

from src.agents.dense_retrieval_agent import DenseRetrieverAgent
from src.agents.query_reformulation_agent import QueryReformulationAgent
from src.agents.analyzer_agent import AnalyzerAgent
from src.agents.synthesizer_agent import SynthesizerAgent
from src.agents.validator_agent import ValidatorAgent
from src.experiments.pyserini_bm25 import retrieve_bm25_pyserini

logger = logging.getLogger(__name__)


def _extract_text_and_usage(out):
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


def run_functional_agents(
    bundle,
    model_cfg: Dict[str, Any],
    strategy_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    top_k: int = 10,
):
    if bundle is None:
        raise ValueError("Strategy 6 requires an IRDatasetBundle.")

    model_cfg = model_cfg or {}
    strategy_cfg = strategy_cfg or {}
    dataset_cfg = dataset_cfg or {}

    bm25_top_k = int(strategy_cfg.get("bm25_top_k", 10))
    dense_top_k = int(strategy_cfg.get("dense_top_k", 10))
    reformulate_k = int(strategy_cfg.get("reformulate_k", 3))
    reform_dense_top_k = int(strategy_cfg.get("reform_dense_top_k", 10))
    gather_top_k_total = int(strategy_cfg.get("gather_top_k_total", 10))

    # NEW: batching for global rerank
    reranker_batch_size = int(strategy_cfg.get("reranker_batch_size", 16))

    # Dataset name resolution
    if "year" in dataset_cfg:
        dataset_name = f"trecdl_{dataset_cfg['year']}"
    elif dataset_cfg.get("pyserini_prebuilt_index"):
        dataset_name = dataset_cfg["pyserini_prebuilt_index"]
    else:
        raise ValueError("dataset_cfg must include 'year' or 'pyserini_prebuilt_index'.")

    safe_name = _safe_dataset_name(dataset_name)
    faiss_path = Path("outputs") / "dense" / f"{safe_name}_judged" / "faiss_index.bin"
    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

    corpus_df = bundle.corpus_df.copy()
    corpus_df["docid"] = corpus_df["docid"].astype(str)
    docid_to_passage = dict(zip(corpus_df["docid"], corpus_df["passage"].astype(str)))

    embedding_model = model_cfg.get("retriever", "BAAI/bge-large-en-v1.5")
    reranker_name = model_cfg.get("reranker", "BAAI/bge-reranker-large")

    dense_retriever = DenseRetrieverAgent(
        df=corpus_df,
        doc_col="passage",
        docid_col="docid",
        model_name=embedding_model,
        index_path=str(faiss_path),
    )

    reformulator = QueryReformulationAgent(
        temperature=float(strategy_cfg.get("reformulator_temperature", 0.2)),
        max_tokens=int(strategy_cfg.get("reformulator_max_tokens", 256)),
    )

    reranker = FlagReranker(reranker_name, use_fp16=True)

    analyzer = AnalyzerAgent()
    synthesizer = SynthesizerAgent()
    validator = ValidatorAgent()

    cols = ["qid", "query"]
    if "complexity" in bundle.queries_df.columns:
        cols.append("complexity")

    queries_df = bundle.queries_df[cols].drop_duplicates().copy()
    queries_df["qid"] = queries_df["qid"].astype(str)

    results: List[Dict[str, Any]] = []

    for _, row in queries_df.iterrows():
        qid = row["qid"]
        query = row["query"]
        complexity = row.get("complexity")

        usage_calls = []

        # -------------------------
        # Gatherer (BM25 + Dense)
        # -------------------------
        one_q_df = pd.DataFrame([{"qid": qid, "query": query}])

        bm25_df = retrieve_bm25_pyserini(
            one_q_df,
            top_k=bm25_top_k,
            index_dir=dataset_cfg.get("pyserini_index_dir"),
            prebuilt_index=dataset_cfg.get("pyserini_prebuilt_index"),
        )

        bm25_docs = []
        for _, hit in bm25_df.iterrows():
            docid = str(hit.get("docid", "")).strip()
            if not docid:
                continue
            passage = hit.get("passage") or docid_to_passage.get(docid, "")
            if not passage:
                continue
            bm25_docs.append(
                {
                    "docid": docid,
                    "passage": passage,
                    "source": "bm25",
                    "bm25_score": float(hit.get("bm25_score", 0.0)),
                }
            )

        dense_hits = dense_retriever.retrieve(query=query, k=dense_top_k)
        dense_docs = [
            {
                "docid": str(d["docid"]),
                "passage": d.get("passage", ""),
                "source": "dense",
                "dense_score": float(d.get("score", 0.0)),
            }
            for d in dense_hits
        ]

        # -------------------------
        # Reformulation
        # -------------------------
        ref_out = reformulator.rewrite_with_usage(query)
        subqueries, u = _extract_text_and_usage(ref_out)

        subqueries = (subqueries or [])[:reformulate_k]
        if u:
            usage_calls.append(u)

        reform_dense_docs = []
        for sq in subqueries:
            for d in dense_retriever.retrieve(query=sq, k=reform_dense_top_k):
                reform_dense_docs.append(
                    {
                        "docid": str(d["docid"]),
                        "passage": d.get("passage", ""),
                        "source": "reform_dense",
                        "subquery": sq,
                        "dense_score": float(d.get("score", 0.0)),
                    }
                )

        # Merge + dedupe (docid-level)
        merged = {d["docid"]: d for d in (bm25_docs + dense_docs + reform_dense_docs)}
        merged_docs = list(merged.values())

        # -------------------------
        # Global rerank (BATCHED)
        # -------------------------
        pairs = [[query, d["passage"]] for d in merged_docs]
        cross_scores: List[float] = []

        bs = max(1, int(reranker_batch_size))
        for start in range(0, len(pairs), bs):
            batch = pairs[start : start + bs]
            out = reranker.compute_score(batch)
            if isinstance(out, (list, tuple, np.ndarray)):
                cross_scores.extend([float(x) for x in out])
            else:
                cross_scores.append(float(out))

        if len(cross_scores) != len(merged_docs):
            raise RuntimeError(
                f"[functional_agents] Reranker returned {len(cross_scores)} scores for {len(merged_docs)} docs."
            )

        scored = []
        for d, s in zip(merged_docs, cross_scores):
            d2 = dict(d)
            d2["cross_score"] = float(s)
            scored.append(d2)

        scored.sort(key=lambda x: x["cross_score"], reverse=True)
        retrieved_docs = scored[:gather_top_k_total]

        # -------------------------
        # Functional agents (collect usage)
        # -------------------------
        ana_out = analyzer.run_with_usage(query, retrieved_docs)
        analysis_text, u = _extract_text_and_usage(ana_out)
        if u:
            usage_calls.append(u)

        syn_out = synthesizer.run_with_usage(query, retrieved_docs, analysis_text)
        draft, u = _extract_text_and_usage(syn_out)
        if u:
            usage_calls.append(u)

        val_out = validator.run_with_usage(query, retrieved_docs, draft)
        final_answer, u = _extract_text_and_usage(val_out)
        if u:
            usage_calls.append(u)

        usage_total = _sum_usage(usage_calls)

        results.append(
            {
                "qid": qid,
                "query": query,
                **({"complexity": complexity} if complexity is not None else {}),
                "subqueries": subqueries,
                "retrieved_docs": retrieved_docs,
                "analysis": analysis_text,
                "final_answer": final_answer,
                "usage": usage_total,
                "usage_calls": usage_calls,
                "usage_total": usage_total,
                "strategy": "functional_agents",
                "metadata": {
                    "dataset": dataset_name,
                    "faiss_index_path": str(faiss_path),
                    "embedding_model": embedding_model,
                    "reranker_model": reranker_name,
                    "bm25_top_k": bm25_top_k,
                    "dense_top_k": dense_top_k,
                    "reformulate_k": reformulate_k,
                    "reform_dense_top_k": reform_dense_top_k,
                    "gather_top_k_total": gather_top_k_total,
                    "reranker_batch_size": reranker_batch_size,
                },
            }
        )

    return results
