import logging
import pandas as pd
from typing import List, Dict, Any

from src.experiments.rag import retrieve_bm25
from src.agents.query_reformulation_agent import QueryReformulationAgent
from src.agents.dense_retrieval_agent import DenseRetrieverAgent  # ✅ use agent-level retriever

logger = logging.getLogger(__name__)


def _dedupe_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for d in docs:
        docid = d.get("docid")
        if not docid or docid in seen:
            continue
        seen.add(docid)
        out.append(d)
    return out


class GathererAgent:
    """
    Collects a diverse doc set via:
      1) BM25
      2) Dense
      3) Reformulate -> Dense (using subqueries)

    Returns merged, deduped docs with a 'source' field.
    """

    def __init__(
        self,
        df_corpus,                 # pass corpus dataframe once (docid, passage)
        bm25_top_k: int = 10,
        dense_top_k: int = 10,
        reformulate_k: int = 3,
        reform_dense_top_k: int = 10,
        dense_model_name: str = "BAAI/bge-large-en-v1.5",
        index_path: str = "outputs/faiss_index.bin",
    ):
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.reformulate_k = reformulate_k
        self.reform_dense_top_k = reform_dense_top_k

        self.reformulator = QueryReformulationAgent()

        # One dense retriever over the whole corpus
        self.retriever = DenseRetrieverAgent(
            df=df_corpus,
            doc_col="passage",
            docid_col="docid",
            model_name=dense_model_name,
            index_path=index_path
        )

    def gather(
        self,
        df,  # original trec_df (qid/query/docid/passage)
        query_row: Dict[str, Any],  # {"qid":..., "query":...}
        model_cfg: dict,
        strategy_cfg: dict,
        dataset_cfg: dict,
    ) -> Dict[str, Any]:
        qid = query_row["qid"]
        query = query_row["query"]

        # --- BM25 ---
        bm25_df_all = dataset_cfg["bm25_df_all"]
        bm25_rows = (
            bm25_df_all[bm25_df_all["qid"] == qid]
            .sort_values("rank")
            .head(self.bm25_top_k)
            .to_dict("records")
        )

        bm25_docs = [
            {
                "docid": r["docid"],
                "passage": r.get("passage", ""),
                "source": "bm25",
                "rank": r.get("rank"),
                "bm25_score": r.get("bm25_score"),
            }
            for r in bm25_rows
        ]


        # --- Dense (direct) ---
        dense_docs_raw = self.retriever.retrieve(query=query, k=self.dense_top_k)
        dense_docs = [
            {
                "docid": d["docid"],
                "passage": d.get("doc_text") or d.get("text") or d.get("passage", ""),
                "source": "dense",
            }
            for d in dense_docs_raw
        ]

        # --- Reformulation -> Dense (NOW uses subquery text) ---
        subqs = self.reformulator.rewrite(query)[: self.reformulate_k]

        reform_docs_all = []
        for sq in subqs:
            docs_sq = self.retriever.retrieve(query=sq, k=self.reform_dense_top_k)
            reform_docs_all.extend([
                {
                    "docid": d["docid"],
                    "passage": d.get("doc_text") or d.get("text") or d.get("passage", ""),
                    "source": f"reform_dense",
                    "subquery": sq,
                }
                for d in docs_sq
            ])

        merged = _dedupe_docs(bm25_docs + dense_docs + reform_docs_all)

        return {
            "qid": qid,
            "query": query,
            "subqueries": subqs,
            "docs": merged
        }
