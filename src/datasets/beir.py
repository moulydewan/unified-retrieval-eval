import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterator, Optional, Tuple, List

import pandas as pd
from src.datasets.datasetbundle import IRDatasetBundle 

logger = logging.getLogger(__name__)

# BEIR imports
from beir.datasets.data_loader import GenericDataLoader
from beir.util import download_and_unzip


@dataclass
class BEIRDoc:
    doc_id: str
    title: str
    text: str


@dataclass
class BEIRQuery:
    query_id: str
    text: str


@dataclass
class BEIRQrel:
    query_id: str
    doc_id: str
    relevance: int


class BEIRSCIDOCSAdapter:
    """Adapter for BEIR SCIDOCS dataset."""

    def __init__(self, data_dir: str = "datasets/beir", split: str = "test"):
        """
        Args:
            data_dir: root folder where BEIR datasets will be stored/downloaded.
            split: one of {"train", "dev", "test"} for qrels/queries split.
        """
        if split not in {"train", "dev", "test"}:
            raise ValueError("split must be one of {'train','dev','test'}")

        self.dataset_name = "scidocs"
        self.split = split

        self.data_root = Path(data_dir).resolve()
        self.dataset_path = self.data_root / self.dataset_name

        self.name = f"beir/{self.dataset_name}:{self.split}"

        # Ensure dataset exists locally (download if missing)
        self._ensure_downloaded()

        logger.info(f"Initializing dataset: {self.name} at {self.dataset_path}")
    
    def load_queries_df(
        self,
        max_items: Optional[int] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Load ONLY queries that have qrels (qrels-backed queries).
        This matches TREC DL + NQ behavior and is safe for subset sampling.
        """
        # Load queries + qrels only (corpus is irrelevant here, but BEIR loader returns it anyway)
        corpus, queries, qrels = GenericDataLoader(str(self.dataset_path)).load(split=self.split)

        qids_with_qrels = sorted([qid for qid in queries.keys() if qid in qrels])

        rows = [(qid, queries[qid]) for qid in qids_with_qrels]
        df = (
            pd.DataFrame(rows, columns=["qid", "query"])
            .drop_duplicates("qid")
            .reset_index(drop=True)
        )

        if max_items is not None and len(df) > max_items:
            df = df.sample(n=max_items, random_state=seed).reset_index(drop=True)

        return df


    def _ensure_downloaded(self) -> None:
        """
        Downloads and unzips SCIDOCS if not present locally.
        """
        if self.dataset_path.exists() and any(self.dataset_path.iterdir()):
            return

        self.data_root.mkdir(parents=True, exist_ok=True)

        # Official BEIR dataset URL pattern
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
        logger.info(f"Downloading {self.dataset_name} from: {url}")
        download_and_unzip(url, str(self.data_root))

        if not self.dataset_path.exists():
            raise RuntimeError(f"Download finished but dataset not found at: {self.dataset_path}")

    # -------------------------------
    # Load into memory
    # -------------------------------
        
    def load(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Load SCIDOCS into memory (JUDGED-ONLY corpus).
        If `limit` is specified, keeps only the first N queries that have qrels.
        Corpus IS subselected by qrels docids.
        """
        try:
            corpus, queries, qrels = GenericDataLoader(str(self.dataset_path)).load(split=self.split)

            # Restrict to queries that actually have qrels
            qids_with_qrels = sorted([qid for qid in queries.keys() if qid in qrels])
            if limit:
                qids_with_qrels = qids_with_qrels[:limit]

            # Filter queries + qrels to the chosen qids
            queries_f = {qid: queries[qid] for qid in qids_with_qrels}
            qrels_f = {qid: qrels[qid] for qid in qids_with_qrels}

            # ---- judged-only corpus: keep only docids that appear in qrels ----
            judged_docids = set()
            for qid, doc_rel_map in qrels_f.items():
                judged_docids.update(doc_rel_map.keys())

            corpus_f = {docid: corpus[docid] for docid in judged_docids if docid in corpus}

            logger.info(
                f"[BEIR {self.dataset_name} | {self.split}] Loaded JUDGED corpus={len(corpus_f)} docs, "
                f"{len(queries_f)} queries, {sum(len(v) for v in qrels_f.values())} qrels."
            )

            return {"corpus": corpus_f, "queries": queries_f, "qrels": qrels_f}

        except Exception as e:
            logger.error(f"Failed to load {self.name}: {e}")
            return {"corpus": {}, "queries": {}, "qrels": {}}


    # -------------------------------
    # Iterators for streaming (optional)
    # -------------------------------
    def iter_docs(self, data: Dict[str, Any], limit: Optional[int] = None) -> Iterator[BEIRDoc]:
        """
        Yield docs from already-loaded `data` to avoid full-corpus iteration.
        """
        corpus = data.get("corpus", {})
        count = 0
        for doc_id, doc in corpus.items():
            yield BEIRDoc(
                doc_id=doc_id,
                title=(doc.get("title") or ""),
                text=(doc.get("text") or ""),
            )
            count += 1
            if limit and count >= limit:
                break

    def iter_queries(self, data: Dict[str, Any], limit: Optional[int] = None) -> Iterator[BEIRQuery]:
        queries = data.get("queries", {})
        count = 0
        for qid, text in queries.items():
            yield BEIRQuery(query_id=qid, text=text)
            count += 1
            if limit and count >= limit:
                break

    def iter_qrels(self, data: Dict[str, Any], limit: Optional[int] = None) -> Iterator[BEIRQrel]:
        qrels = data.get("qrels", {})
        count = 0
        for qid, doc_rel_map in qrels.items():
            for doc_id, rel in doc_rel_map.items():
                yield BEIRQrel(query_id=qid, doc_id=doc_id, relevance=int(rel))
                count += 1
                if limit and count >= limit:
                    return

    # -------------------------------
    # Convert to flat DataFrame
    # -------------------------------
    def beir_df(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert loaded BEIR data into a flat DataFrame:
        [qid, query, docid, title, text, rel]
        """
        if not all(k in data for k in ["corpus", "queries", "qrels"]):
            raise ValueError("Input must contain 'corpus', 'queries', and 'qrels' keys.")

        corpus = data["corpus"]
        queries = data["queries"]
        qrels = data["qrels"]

        rows: List[Tuple[str, str, str, str, str, int]] = []

        for qid, doc_rel_map in qrels.items():
            qtext = queries.get(qid)
            if not qtext:
                continue

            for docid, rel in doc_rel_map.items():
                doc = corpus.get(docid)
                if not doc:
                    continue
                title = doc.get("title") or ""
                text = doc.get("text") or ""
                rows.append((qid, qtext, docid, title, text, int(rel)))

        return pd.DataFrame(rows, columns=["qid", "query", "docid", "title", "text", "rel"])
    
    # -------------------------------
    # NEW: Convert to IRDatasetBundle
    # -------------------------------
    def to_bundle(self, data: Dict[str, Any]) -> IRDatasetBundle:
        """
        Convert loaded BEIR data into the universal bundle:
          - corpus_df:  docid, passage
          - queries_df: qid, query
          - qrels_df:   qid, docid, rel
        """
        if not all(k in data for k in ["corpus", "queries", "qrels"]):
            raise ValueError("Input must contain 'corpus', 'queries', and 'qrels' keys.")

        corpus = data["corpus"]
        queries = data["queries"]
        qrels = data["qrels"]

        # 1 row per doc
        corpus_rows = []
        for docid, doc in corpus.items():
            title = (doc.get("title") or "").strip()
            text = (doc.get("text") or "").strip()

            # Combine into a single "passage" field for retrievers
            passage = f"{title}\n\n{text}".strip() if title else text

            corpus_rows.append((docid, passage))

        corpus_df = pd.DataFrame(corpus_rows, columns=["docid", "passage"])

        # 1 row per query
        queries_df = (
            pd.DataFrame([(qid, qtext) for qid, qtext in queries.items()],
                         columns=["qid", "query"])
            .reset_index(drop=True)
        )

        # 1 row per qrel judgment
        qrels_rows = []
        for qid, doc_rel_map in qrels.items():
            for docid, rel in doc_rel_map.items():
                qrels_rows.append((qid, docid, int(rel)))

        qrels_df = pd.DataFrame(qrels_rows, columns=["qid", "docid", "rel"])

        return IRDatasetBundle(
            corpus_df=corpus_df,
            queries_df=queries_df,
            qrels_df=qrels_df,
        )

    def load_bundle(self, limit: Optional[int] = None) -> IRDatasetBundle:
        """Convenience: load + convert to bundle."""
        data = self.load(limit=limit)
        return self.to_bundle(data)
