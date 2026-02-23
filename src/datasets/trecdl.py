import logging
import ir_datasets
import pandas as pd
from typing import Dict, Any, Iterator, Optional

from src.datasets.datasetbundle import IRDatasetBundle

logger = logging.getLogger(__name__)


class TRECDLAdapter:
    """Adapter for TREC Deep Learning 2019–2021 datasets (MSMARCO corpus)."""

    YEARS = [2019, 2020, 2021]
    MODES = ["passage", "document"]

    def __init__(self, year: int = 2020, mode: str = "passage"):
        if year not in self.YEARS:
            raise ValueError(f"Supported years: {self.YEARS}")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of: {self.MODES}")

        # Dataset ID (ir_datasets naming scheme)
        if year in [2019, 2020]:
            dataset_id = f"msmarco-{mode}/trec-dl-{year}"
        else:  # 2021
            dataset_id = f"msmarco-{mode}-v2/trec-dl-{year}"

        logger.info(f"Initializing dataset: {dataset_id}")

        try:
            self.dataset = ir_datasets.load(dataset_id)
        except KeyError:
            raise ValueError(f"Dataset not found in ir_datasets: {dataset_id}")

        self.name = dataset_id
        self.year = year
        self.mode = mode

    # -------------------------------
    # Load ONLY qrels-backed queries
    # -------------------------------
    def load_queries_df(self, max_items: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
        qids_with_qrels = set(r.query_id for r in self.dataset.qrels_iter())

        rows = []
        for q in self.dataset.queries_iter():
            if q.query_id in qids_with_qrels:
                rows.append((q.query_id, getattr(q, "text", "")))

        df = (
            pd.DataFrame(rows, columns=["qid", "query"])
            .drop_duplicates("qid")
            .reset_index(drop=True)
        )

        if max_items is not None and len(df) > max_items:
            df = df.sample(n=max_items, random_state=seed).reset_index(drop=True)

        return df

    # -------------------------------
    # Load judged-only (ALWAYS)
    # -------------------------------
    def load(self, limit: Optional[int] = None, full_corpus: bool = False) -> Dict[str, Any]:
        """
        Always loads JUDGED-ONLY docs (docids appearing in qrels for selected queries).

        NOTE: `full_corpus` is intentionally ignored for TREC DL because MSMARCO is too large
        to materialize in-memory, and retrieval uses Pyserini for BM25.
        """
        try:
            # Step 1: Build mapping of query → qrels (single pass)
            qrel_map: Dict[str, list] = {}
            for r in self.dataset.qrels_iter():
                qrel_map.setdefault(r.query_id, []).append(r)

            # Step 2: Restrict to queries that have qrels
            all_queries = [q for q in self.dataset.queries_iter() if q.query_id in qrel_map]
            if limit:
                all_queries = all_queries[:limit]

            # Step 3: Gather qrels for these queries
            qrels = [r for q in all_queries for r in qrel_map.get(q.query_id, [])]

            # Step 4: Materialize judged docs only
            docids = {r.doc_id for r in qrels}
            store = self.dataset.docs_store()
            docs = []
            for did in docids:
                d = store.get(did)
                if d is not None:
                    docs.append(d)

            logger.info(
                f"[TREC DL {self.year}] Loaded JUDGED corpus={len(docs)} docs, "
                f"{len(all_queries)} queries, {len(qrels)} qrels."
            )

            # full_corpus is forced False to prevent empty corpus_df paths downstream
            return {"docs": docs, "queries": all_queries, "qrels": qrels, "full_corpus": False}

        except Exception as e:
            logger.error(f"Failed to load {self.name}: {e}")
            return {"docs": [], "queries": [], "qrels": [], "full_corpus": False}

    # -------------------------------
    # Iterators for streaming (optional)
    # -------------------------------
    def iter_docs(self, limit: Optional[int] = None) -> Iterator[Any]:
        count = 0
        for doc in self.dataset.docs_iter():
            yield doc
            count += 1
            if limit and count >= limit:
                break

    def iter_queries(self, limit: Optional[int] = None) -> Iterator[Any]:
        count = 0
        for q in self.dataset.queries_iter():
            yield q
            count += 1
            if limit and count >= limit:
                break

    def iter_qrels(self, limit: Optional[int] = None) -> Iterator[Any]:
        count = 0
        for r in self.dataset.qrels_iter():
            yield r
            count += 1
            if limit and count >= limit:
                break

    # -------------------------------
    # Convert to flat DataFrame
    # -------------------------------
    def trec_df(self, data: Dict[str, Any]) -> pd.DataFrame:
        if not all(k in data for k in ["docs", "queries", "qrels"]):
            raise ValueError("Input must contain 'docs', 'queries', and 'qrels' keys.")

        doc_lookup = {d.doc_id: getattr(d, "text", "") for d in data["docs"]}
        query_lookup = {q.query_id: getattr(q, "text", "") for q in data["queries"]}

        rows = []
        for r in data["qrels"]:
            qid = r.query_id
            docid = r.doc_id
            rel = getattr(r, "relevance", 0)
            query = query_lookup.get(qid)
            passage = doc_lookup.get(docid)
            if query and passage:
                rows.append((qid, query, docid, passage, rel))

        return pd.DataFrame(rows, columns=["qid", "query", "docid", "passage", "rel"])

    # -------------------------------
    # Convert to IRDatasetBundle (ALWAYS judged-only)
    # -------------------------------
    def to_bundle(self, data: Dict[str, Any]) -> IRDatasetBundle:
        """
        Always returns a non-empty judged-only corpus_df (unless dataset has no qrels/docs).
        Removes the prior 'full_corpus=True => empty corpus_df' behavior.
        """
        df = self.trec_df(data)

        queries_df = df[["qid", "query"]].drop_duplicates("qid").reset_index(drop=True)
        corpus_df = df[["docid", "passage"]].drop_duplicates("docid").reset_index(drop=True)
        qrels_df = df[["qid", "docid", "rel"]].drop_duplicates().reset_index(drop=True)

        return IRDatasetBundle(corpus_df=corpus_df, queries_df=queries_df, qrels_df=qrels_df)

    def load_bundle(self, limit: Optional[int] = None, full_corpus: bool = False) -> IRDatasetBundle:
        data = self.load(limit=limit, full_corpus=full_corpus)
        return self.to_bundle(data)
