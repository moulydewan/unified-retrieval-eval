# src/agents/dense_retrieval_agent.py
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import faiss
from FlagEmbedding import FlagModel, FlagReranker
from tqdm import tqdm

logger = logging.getLogger(__name__)

# -------------------------
# In-process model caches (match Strategy 2)
# -------------------------
_MODEL_CACHE: Dict[str, FlagModel] = {}
_RERANKER_CACHE: Dict[str, FlagReranker] = {}


def get_flag_model(model_name: str) -> FlagModel:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = FlagModel(
            model_name,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            use_fp16=True,
        )
    return _MODEL_CACHE[model_name]


def get_flag_reranker(model_name: str) -> FlagReranker:
    if model_name not in _RERANKER_CACHE:
        _RERANKER_CACHE[model_name] = FlagReranker(model_name, use_fp16=True)
    return _RERANKER_CACHE[model_name]


class DenseRetrieverAgent:
    def __init__(
        self,
        df: pd.DataFrame,
        doc_col: str = "passage",
        docid_col: str = "docid",
        model_name: str = "BAAI/bge-large-en-v1.5",
        index_path: str | None = None,
        reranker_name: str = "BAAI/bge-reranker-large",
        use_reranker: bool = True,
        reranker_batch_size: int = 16,
        show_rerank_progress: bool = False,
    ):
        # ---- deterministic order (must match Strategy 2) ----
        df = df.copy()
        df[docid_col] = df[docid_col].astype(str)
        df = df.sort_values(docid_col).reset_index(drop=True)

        self.df = df
        self.doc_col = doc_col
        self.docid_col = docid_col
        self.passages = self.df[self.doc_col].astype(str).tolist()

        logger.info(f"[DenseRetrieverAgent] Loading bi-encoder: {model_name}")
        self.model = get_flag_model(model_name)

        if index_path is None:
            raise ValueError("DenseRetrieverAgent requires index_path (Strategy 4 uses Strategy 2 cache).")

        self.index_path = Path(index_path)
        self.cache_dir = self.index_path.parent
        self.emb_path = self.cache_dir / "doc_embeddings.npy"
        self.docids_path = self.cache_dir / "docids.npy"

        # ---- validate cache belongs to *this* corpus order ----
        self._ensure_cache_matches_corpus()
        

        # embeddings + index in SAME DIR as Strategy 2
        self.doc_embeddings = self._load_or_build_embeddings()
        self.index = self._load_or_build_faiss()

        # cross-encoder reranker (batched)
        self.use_reranker = bool(use_reranker)
        self.reranker_name = reranker_name
        self.reranker: Optional[FlagReranker] = None
        self.reranker_batch_size = int(max(1, reranker_batch_size))
        self.show_rerank_progress = bool(show_rerank_progress)

        if self.use_reranker:
            logger.info(f"[DenseRetrieverAgent] Loading cross-encoder reranker: {reranker_name}")
            self.reranker = get_flag_reranker(reranker_name)

    # -------------------------
    # Cache consistency guard
    # -------------------------
    def _ensure_cache_matches_corpus(self) -> None:
        current_docids = self.df[self.docid_col].astype(str).tolist()

        def docid_signature_matches() -> bool:
            if not self.docids_path.exists():
                return False
            try:
                cached = np.load(self.docids_path, allow_pickle=True).tolist()
                cached = [str(x) for x in cached]
                return cached == current_docids
            except Exception:
                return False

        cache_exists = self.emb_path.exists() or self.index_path.exists()

        # 1) DocID order mismatch => delete stale cache
        if cache_exists and not docid_signature_matches():
            logger.warning(
                f"[DenseRetrieverAgent] Cache docid-order mismatch in {self.cache_dir}. "
                "Deleting stale embeddings/index."
            )
            for p in (self.emb_path, self.index_path, self.docids_path):
                if p.exists():
                    p.unlink()

        # 2) FAISS corruption / size mismatch guard
        if self.index_path.exists():
            try:
                idx = faiss.read_index(str(self.index_path))
                if idx.ntotal != len(current_docids):
                    logger.warning(
                        f"[DenseRetrieverAgent] FAISS ntotal mismatch "
                        f"({idx.ntotal} != {len(current_docids)}). Rebuilding."
                    )
                    for p in (self.emb_path, self.index_path, self.docids_path):
                        if p.exists():
                            p.unlink()
            except Exception as e:
                logger.warning(f"[DenseRetrieverAgent] Failed to read FAISS index ({e}). Rebuilding.")
                for p in (self.emb_path, self.index_path, self.docids_path):
                    if p.exists():
                        p.unlink()
        # 3) Ensure docid signature exists (or refresh it)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.docids_path, np.array(current_docids, dtype=object))
        return

    # -------------------------
    # Embeddings (match Strategy 2 behavior)
    # -------------------------
    def _load_or_build_embeddings(self) -> np.ndarray:
        """
        Loads cached embeddings if possible.
        Rebuilds embeddings (and invalidates FAISS) if cache is missing/corrupt
        or size mismatches current corpus.

        IMPORTANT:
          - We store embeddings L2-normalized on disk (Strategy 2 behavior).
          - If an old cache is detected (unnormalized), we upgrade it once and delete FAISS.
        """
        current_docids = self.df[self.docid_col].astype(str).tolist()

        def save_signature() -> None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.docids_path, np.array(current_docids, dtype=object))

        def rebuild() -> np.ndarray:
            # old index points to old embeddings
            if self.index_path.exists():
                self.index_path.unlink()
            emb = self.model.encode_corpus(self.passages).astype(np.float32, copy=False)
            faiss.normalize_L2(emb)  # normalize ONCE, then cache
            np.save(self.emb_path, emb)
            save_signature()
            return emb

        if self.emb_path.exists():
            logger.info(f"[DenseRetrieverAgent] Loading cached embeddings: {self.emb_path}")
            try:
                doc_embeddings = np.load(self.emb_path, mmap_mode="r").astype(np.float32, copy=False)
            except Exception as e:
                logger.warning(f"[DenseRetrieverAgent] Failed to load embeddings ({e}). Rebuilding.")
                doc_embeddings = rebuild()
            else:
                # One-time upgrade: old caches may have been saved unnormalized
                norms = np.linalg.norm(
                    doc_embeddings[: min(1000, len(doc_embeddings))], axis=1
                )
                if not (0.9 <= float(norms.mean()) <= 1.1):
                    logger.info("[DenseRetrieverAgent] Upgrading cached embeddings to L2-normalized format.")
                    doc_embeddings = np.array(doc_embeddings, dtype=np.float32, copy=True)
                    faiss.normalize_L2(doc_embeddings)
                    np.save(self.emb_path, doc_embeddings)
                    # IMPORTANT: FAISS index contains old vectors -> must rebuild
                    if self.index_path.exists():
                        self.index_path.unlink()
                    doc_embeddings = np.load(self.emb_path, mmap_mode="r").astype(np.float32, copy=False)


                # Size mismatch => rebuild embeddings (+ FAISS invalidated in rebuild())
                if doc_embeddings.shape[0] != len(self.passages):
                    logger.warning(
                        f"[DenseRetrieverAgent] Embedding size mismatch "
                        f"({doc_embeddings.shape[0]} != {len(self.passages)}). Rebuilding embeddings + FAISS."
                    )
                    doc_embeddings = rebuild()
                else:
                    # embeddings OK → ensure signature exists
                    if not self.docids_path.exists():
                        save_signature()
        else:
            logger.info("[DenseRetrieverAgent] Building new corpus embeddings...")
            doc_embeddings = rebuild()

        return doc_embeddings

    # -------------------------
    # FAISS
    # -------------------------
    def _load_or_build_faiss(self):
        def build():
            logger.info("[DenseRetrieverAgent] Building FAISS index...")
            dim = int(self.doc_embeddings.shape[1])
            index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
            index.add(self.doc_embeddings.astype(np.float32, copy=False))
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(self.index_path))
            logger.info(f"[DenseRetrieverAgent] Saved FAISS index to {self.index_path}")
            return index

        if self.index_path.exists():
            logger.info(f"[DenseRetrieverAgent] Loading cached FAISS index: {self.index_path}")
            try:
                index = faiss.read_index(str(self.index_path))
                if index.ntotal != int(self.doc_embeddings.shape[0]):
                    logger.warning(
                        f"[DenseRetrieverAgent] FAISS ntotal mismatch "
                        f"({index.ntotal} != {self.doc_embeddings.shape[0]}). Rebuilding."
                    )
                    self.index_path.unlink()
                    return build()
                return index
            except Exception as e:
                logger.warning(f"[DenseRetrieverAgent] Failed to read FAISS index ({e}). Rebuilding.")
                if self.index_path.exists():
                    self.index_path.unlink()
                return build()

        return build()

    # -------------------------
    # Reranking (batched)
    # -------------------------
    def _rerank_scores_batched(self, query: str, passages: List[str]) -> List[float]:
        assert self.reranker is not None

        pairs = [[query, p] for p in passages]
        scores: List[float] = []

        bs = self.reranker_batch_size
        it = range(0, len(pairs), bs)
        if self.show_rerank_progress and len(pairs) > bs:
            it = tqdm(list(it), desc="[DenseRetrieverAgent] Reranking", leave=False)

        for start in it:
            batch = pairs[start : start + bs]
            out = self.reranker.compute_score(batch)
            if isinstance(out, (list, tuple, np.ndarray)):
                scores.extend([float(x) for x in out])
            else:
                scores.append(float(out))

        if len(scores) != len(passages):
            raise RuntimeError(
                f"[DenseRetrieverAgent] Reranker returned {len(scores)} scores for {len(passages)} passages."
            )
        return scores

    def retrieve(
        self,
        query: str,
        k: int = 10,
        top_k_dense: int = 50,
    ) -> List[Dict[str, Any]]:
        if not query:
            return []

        q = self.model.encode_queries([str(query)]).astype(np.float32)
        faiss.normalize_L2(q)

        top_k_dense = min(int(top_k_dense), len(self.df))
        if top_k_dense <= 0:
            return []

        D, I = self.index.search(q, top_k_dense)

        cand = []
        for dense_rank, (idx, dense_score) in enumerate(zip(I[0], D[0]), start=1):
            if 0 <= idx < len(self.df):
                cand.append(
                    {
                        "idx": int(idx),
                        "docid": self.df.iloc[idx][self.docid_col],
                        "passage": self.passages[idx],
                        "dense_score": float(dense_score),
                        "dense_rank": int(dense_rank),
                    }
                )

        if not cand:
            return []

        if self.use_reranker and self.reranker is not None:
            passages = [c["passage"] for c in cand]
            cross_scores = self._rerank_scores_batched(query, passages)

            for c, s in zip(cand, cross_scores):
                c["cross_score"] = float(s)

            cand.sort(key=lambda x: x["cross_score"], reverse=True)
            out = cand[: min(int(k), len(cand))]

            results = []
            for rerank, c in enumerate(out, start=1):
                results.append(
                    {
                        "query": query,
                        "docid": c["docid"],
                        "passage": c["passage"],
                        "score": c["cross_score"],
                        "cross_score": c["cross_score"],
                        "dense_score": c["dense_score"],
                        "dense_rank": c["dense_rank"],
                        "rerank": int(rerank),
                    }
                )
            return results

        # No reranker fallback
        cand = cand[: min(int(k), len(cand))]
        results = []
        for rank, c in enumerate(cand, start=1):
            results.append(
                {
                    "query": query,
                    "docid": c["docid"],
                    "passage": c["passage"],
                    "score": c["dense_score"],
                    "dense_score": c["dense_score"],
                    "dense_rank": int(rank),
                }
            )
        return results
