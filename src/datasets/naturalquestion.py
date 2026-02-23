# src/datasets/naturalquestion.py
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
from beir.datasets.data_loader import GenericDataLoader
from beir.util import download_and_unzip

from src.datasets.datasetbundle import IRDatasetBundle

logger = logging.getLogger(__name__)


class NaturalQuestionsAdapter:
    """
    Natural Questions adapter using BEIR's 'nq' dataset (IR-format).
    - Same behavior as your BEIRSCIDOCSAdapter, but fixed dataset_name='nq'.
    - load() returns dict with keys: corpus, queries, qrels
    - to_bundle() returns IRDatasetBundle(corpus_df, queries_df, qrels_df)
    """

    def __init__(self, data_dir: str = "datasets/beir", split: str = "test"):
        if split not in {"train", "dev", "test"}:
            raise ValueError("split must be one of {'train','dev','test'}")

        self.dataset_name = "nq"
        self.split = split

        self.data_root = Path(data_dir).resolve()
        self.dataset_path = self.data_root / self.dataset_name

        self.name = f"beir/{self.dataset_name}:{self.split}"

        self._ensure_downloaded()
        logger.info(f"Initializing dataset: {self.name} at {self.dataset_path}")

    def _ensure_downloaded(self) -> None:
        if self.dataset_path.exists() and any(self.dataset_path.iterdir()):
            return

        self.data_root.mkdir(parents=True, exist_ok=True)

        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
        logger.info(f"Downloading {self.dataset_name} from: {url}")
        download_and_unzip(url, str(self.data_root))

        if not self.dataset_path.exists():
            raise RuntimeError(f"Download finished but dataset not found at: {self.dataset_path}")

    def load_queries_df(self, max_items: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
        """
        Load ONLY queries that have qrels (qrels-backed queries).
        Safe for subset sampling.
        """
        _corpus, queries, qrels = GenericDataLoader(str(self.dataset_path)).load(split=self.split)

        qids_with_qrels = sorted([qid for qid in queries.keys() if qid in qrels])

        rows = [(str(qid), str(queries[qid])) for qid in qids_with_qrels]
        df = pd.DataFrame(rows, columns=["qid", "query"]).drop_duplicates("qid").reset_index(drop=True)

        if max_items is not None and len(df) > max_items:
            df = df.sample(n=max_items, random_state=seed).reset_index(drop=True)

        return df
    
    def load(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Load BEIR NQ into memory (JUDGED-ONLY corpus).
        If `limit` is specified, keeps only the first N qrels-backed queries.
        Corpus IS subselected by qrels docids.
        """
        try:
            corpus, queries, qrels = GenericDataLoader(str(self.dataset_path)).load(split=self.split)

            qids_with_qrels = sorted([qid for qid in queries.keys() if qid in qrels])
            if limit:
                qids_with_qrels = qids_with_qrels[:limit]

            queries_f = {str(qid): str(queries[qid]) for qid in qids_with_qrels}
            qrels_f = {str(qid): qrels[qid] for qid in qids_with_qrels}

            # ---- judged-only corpus ----
            judged_docids = set()
            for qid, doc_rel_map in qrels_f.items():
                judged_docids.update(doc_rel_map.keys())

            corpus_f = {str(docid): corpus[docid] for docid in judged_docids if docid in corpus}

            logger.info(
                f"[BEIR NQ | {self.split}] Loaded JUDGED corpus={len(corpus_f)} docs, "
                f"{len(queries_f)} queries, {sum(len(v) for v in qrels_f.values())} qrels."
            )
            return {"corpus": corpus_f, "queries": queries_f, "qrels": qrels_f}

        except Exception as e:
            logger.error(f"Failed to load {self.name}: {e}")
            return {"corpus": {}, "queries": {}, "qrels": {}}


    def to_bundle(self, data: Dict[str, Any]) -> IRDatasetBundle:
        """
        Convert loaded BEIR NQ data into IRDatasetBundle:
          - corpus_df:  docid, passage
          - queries_df: qid, query
          - qrels_df:   qid, docid, rel
        """
        if not all(k in data for k in ["corpus", "queries", "qrels"]):
            raise ValueError("Input must contain 'corpus', 'queries', and 'qrels' keys.")

        corpus = data["corpus"]
        queries = data["queries"]
        qrels = data["qrels"]

        # corpus.jsonl has: {"_id": ..., "title": ..., "text": ...}
        corpus_rows: List[Tuple[str, str]] = []
        for docid, doc in corpus.items():
            title = (doc.get("title") or "").strip()
            text = (doc.get("text") or "").strip()
            passage = f"{title}\n\n{text}".strip() if title else text
            corpus_rows.append((str(docid), passage))

        corpus_df = pd.DataFrame(corpus_rows, columns=["docid", "passage"])

        queries_df = (
            pd.DataFrame([(str(qid), str(qtext)) for qid, qtext in queries.items()], columns=["qid", "query"])
            .reset_index(drop=True)
        )

        qrels_rows: List[Tuple[str, str, int]] = []
        for qid, doc_rel_map in qrels.items():
            for docid, rel in doc_rel_map.items():
                qrels_rows.append((str(qid), str(docid), int(rel)))

        qrels_df = pd.DataFrame(qrels_rows, columns=["qid", "docid", "rel"])

        return IRDatasetBundle(corpus_df=corpus_df, queries_df=queries_df, qrels_df=qrels_df)

    def load_bundle(self, limit: Optional[int] = None) -> IRDatasetBundle:
        data = self.load(limit=limit)
        return self.to_bundle(data)
