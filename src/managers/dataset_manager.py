# src/managers/dataset_manager.py
from __future__ import annotations

from typing import Tuple, Optional, Dict, Any
import pandas as pd

from src.datasets.trecdl import TRECDLAdapter
from src.datasets.beir import BEIRSCIDOCSAdapter
from src.datasets.naturalquestion import NaturalQuestionsAdapter
from src.datasets.datasetbundle import IRDatasetBundle


def _stable_sort_corpus(bundle: IRDatasetBundle) -> None:
    """
    Ensure deterministic corpus row order so FAISS index row i always maps to corpus_df row i.
    This is critical when reusing cached doc_embeddings.npy / faiss_index.bin across runs/strategies.
    """
    if bundle is None or getattr(bundle, "corpus_df", None) is None:
        return
    if "docid" not in bundle.corpus_df.columns:
        return
    bundle.corpus_df["docid"] = bundle.corpus_df["docid"].astype(str)
    bundle.corpus_df = bundle.corpus_df.sort_values("docid").reset_index(drop=True)


class DatasetManager:
    def __init__(self):
        self.registry = {
            "trecdl": self.load_trec_dl,
            "trecdl_2019": lambda **kwargs: self.load_trec_dl(kwargs.get("cfg") or {}, kwargs.get("limit"), 2019),
            "trecdl_2020": lambda **kwargs: self.load_trec_dl(kwargs.get("cfg") or {}, kwargs.get("limit"), 2020),
            "trecdl_2021": lambda **kwargs: self.load_trec_dl(kwargs.get("cfg") or {}, kwargs.get("limit"), 2021),
            "beir": self.load_beir_scidocs,
            "nq": self.load_beir_nq,
        }

    def load_bundle(
        self,
        name: str,
        cfg: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Tuple[IRDatasetBundle, str]:
        if name not in self.registry:
            raise ValueError(f"Dataset '{name}' not supported. Supported: {list(self.registry.keys())}")

        cfg = cfg or {}
        return self.registry[name](cfg=cfg, limit=limit)

    # ---------------------------
    # Dataset loaders
    # ---------------------------
    def load_trec_dl(self, cfg: Dict[str, Any], limit: Optional[int], year: Optional[int]):
        chosen_year = year or cfg.get("year", 2020)
        mode = cfg.get("mode", "passage")

        adapter = TRECDLAdapter(year=int(chosen_year), mode=mode)

        bundle = adapter.load_bundle(limit=None, full_corpus=False)

        # ---- IMPORTANT: deterministic corpus order for FAISS caching ----
        _stable_sort_corpus(bundle)

        bundle.queries_df["qid"] = bundle.queries_df["qid"].astype(str)
        subset = pd.read_json(
            f"subsets/trec_dl_{chosen_year}_50.jsonl", lines=True
        )[["qid", "complexity"]]
        subset["qid"] = subset["qid"].astype(str)

        bundle.queries_df = bundle.queries_df.merge(subset, on="qid", how="left")
        bundle.queries_df = bundle.queries_df[
            bundle.queries_df["complexity"].notna()
        ].reset_index(drop=True)

        # LIMIT to work in CLI
        if limit is not None:
            bundle.queries_df = bundle.queries_df.head(int(limit)).reset_index(drop=True)

        # Keep qrels aligned with filtered queries
        keep_qids = set(bundle.queries_df["qid"].astype(str))
        bundle.qrels_df["qid"] = bundle.qrels_df["qid"].astype(str)
        bundle.qrels_df = bundle.qrels_df[bundle.qrels_df["qid"].isin(keep_qids)].reset_index(drop=True)

        return bundle, str(chosen_year)

    def load_beir_scidocs(self, cfg: Dict[str, Any], limit: Optional[int]):
        adapter = BEIRSCIDOCSAdapter(
            data_dir=cfg.get("data_dir", "datasets/beir"),
            split=cfg.get("split", "test"),
        )

        bundle = adapter.load_bundle(limit=None)

        # ---- IMPORTANT: deterministic corpus order for FAISS caching ----
        _stable_sort_corpus(bundle)

        bundle.queries_df["qid"] = bundle.queries_df["qid"].astype(str)
        subset = pd.read_json(
            "subsets/beir_scidocs_50.jsonl", lines=True
        )[["qid", "complexity"]]
        subset["qid"] = subset["qid"].astype(str)

        bundle.queries_df = bundle.queries_df.merge(subset, on="qid", how="left")
        bundle.queries_df = bundle.queries_df[
            bundle.queries_df["complexity"].notna()
        ].reset_index(drop=True)

        # LIMIT to work in CLI
        if limit is not None:
            bundle.queries_df = bundle.queries_df.head(int(limit)).reset_index(drop=True)

        # Keep qrels aligned with filtered queries
        keep_qids = set(bundle.queries_df["qid"].astype(str))
        bundle.qrels_df["qid"] = bundle.qrels_df["qid"].astype(str)
        bundle.qrels_df = bundle.qrels_df[bundle.qrels_df["qid"].isin(keep_qids)].reset_index(drop=True)

        return bundle, ""

    def load_beir_nq(self, cfg: Dict[str, Any], limit: Optional[int]):
        adapter = NaturalQuestionsAdapter(
            data_dir=cfg.get("data_dir", "datasets/beir"),
            split=cfg.get("split", "test"),
        )

        bundle = adapter.load_bundle(limit=None)

        # ---- IMPORTANT: deterministic corpus order for FAISS caching ----
        _stable_sort_corpus(bundle)

        bundle.queries_df["qid"] = bundle.queries_df["qid"].astype(str)
        subset = pd.read_json(
            "subsets/beir_nq_50.jsonl", lines=True
        )[["qid", "complexity"]]
        subset["qid"] = subset["qid"].astype(str)

        bundle.queries_df = bundle.queries_df.merge(subset, on="qid", how="left")
        bundle.queries_df = bundle.queries_df[
            bundle.queries_df["complexity"].notna()
        ].reset_index(drop=True)

        # LIMIT to work in CLI
        if limit is not None:
            bundle.queries_df = bundle.queries_df.head(int(limit)).reset_index(drop=True)

        # Keep qrels aligned with filtered queries
        keep_qids = set(bundle.queries_df["qid"].astype(str))
        bundle.qrels_df["qid"] = bundle.qrels_df["qid"].astype(str)
        bundle.qrels_df = bundle.qrels_df[bundle.qrels_df["qid"].isin(keep_qids)].reset_index(drop=True)

        return bundle, ""
