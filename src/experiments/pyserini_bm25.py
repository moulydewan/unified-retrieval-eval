# src/experiments/pyserini_bm25.py
import json
from typing import Optional, List, Dict, Any

import pandas as pd
from pyserini.search.lucene import LuceneSearcher


def _extract_passage_from_raw(raw: str) -> str:
    if raw is None:
        return ""
    raw = str(raw)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return (
                obj.get("contents")
                or obj.get("text")
                or obj.get("body")
                or obj.get("abstract")
                or obj.get("document")
                or raw
            )
    except Exception:
        pass
    return raw


class PyseriniBM25Retriever:
    def __init__(
        self,
        *,
        index_dir: Optional[str] = None,
        prebuilt_index: Optional[str] = None,
        k1: float = 0.9,
        b: float = 0.4,
    ):
        if bool(index_dir) == bool(prebuilt_index):
            raise ValueError("Provide exactly one of index_dir or prebuilt_index.")
        self.searcher = (
            LuceneSearcher.from_prebuilt_index(prebuilt_index)
            if prebuilt_index
            else LuceneSearcher(index_dir)
        )
        try:
            self.searcher.set_bm25(k1=k1, b=b)
        except Exception:
            pass

    def retrieve(self, queries_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for _, r in queries_df.iterrows():
            qid = str(r["qid"])
            query = str(r["query"])
            hits = self.searcher.search(query, k=top_k)

            for rank, hit in enumerate(hits, start=1):
                docid = getattr(hit, "docid", "") or ""
                score = float(getattr(hit, "score", 0.0))
                doc = self.searcher.doc(docid)
                passage = _extract_passage_from_raw(doc.raw()) if doc is not None else ""
                rows.append(
                    {"qid": qid, "query": query, "docid": docid, "passage": passage,
                     "rank": rank, "bm25_score": score}
                )
        return pd.DataFrame(rows)


def retrieve_bm25_pyserini(
    queries_df: pd.DataFrame,
    *,
    top_k: int,
    index_dir: Optional[str] = None,
    prebuilt_index: Optional[str] = None,
    k1: float = 0.9,
    b: float = 0.4,
) -> pd.DataFrame:
    retriever = PyseriniBM25Retriever(
        index_dir=index_dir,
        prebuilt_index=prebuilt_index,
        k1=k1,
        b=b,
    )
    return retriever.retrieve(queries_df, top_k=top_k)
