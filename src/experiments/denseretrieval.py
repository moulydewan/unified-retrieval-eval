# src/experiments/denseretriever.py
import logging
from pathlib import Path
from typing import Dict, Any

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from FlagEmbedding import FlagModel, FlagReranker

from src.datasets.datasetbundle import IRDatasetBundle

logger = logging.getLogger(__name__)

_MODEL_CACHE = {}
_RERANKER_CACHE = {}

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

# ---------------------------
# Dataset-specific cache paths (no splits)
# ---------------------------
def _dense_cache_dir(dataset_name: str, corpus_mode: str) -> Path:
    safe = str(dataset_name).replace("/", "_").replace(":", "_")
    return Path("outputs") / "dense" / f"{safe}_{corpus_mode}"


def _embedding_path(dataset_name: str, corpus_mode: str) -> Path:
    d = _dense_cache_dir(dataset_name, corpus_mode)
    d.mkdir(parents=True, exist_ok=True)
    return d / "doc_embeddings.npy"


def _faiss_path(dataset_name: str, corpus_mode: str) -> Path:
    d = _dense_cache_dir(dataset_name, corpus_mode)
    d.mkdir(parents=True, exist_ok=True)
    return d / "faiss_index.bin"


def _docids_path(dataset_name: str, corpus_mode: str) -> Path:
    d = _dense_cache_dir(dataset_name, corpus_mode)
    d.mkdir(parents=True, exist_ok=True)
    return d / "docids.npy"


# ---------------------------
# FAISS (UNCHANGED behavior)
# ---------------------------
def load_faiss(doc_embeddings: np.ndarray, dim: int, index_path: str | Path):
    index_path = Path(index_path)

    def build() -> faiss.Index:
        index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
        index.add(doc_embeddings)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        logger.info(f"FAISS index saved to {index_path}")
        return index

    if index_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            if index.ntotal != int(doc_embeddings.shape[0]):
                logger.warning(
                    f"[load_faiss] FAISS ntotal mismatch ({index.ntotal} != {doc_embeddings.shape[0]}). Rebuilding."
                )
                index_path.unlink()
                return build()
            return index
        except Exception as e:
            logger.warning(f"[load_faiss] Failed to read FAISS index ({e}). Rebuilding.")
            if index_path.exists():
                index_path.unlink()
            return build()

    return build()



# ---------------------------
# Dense retrieval (BGE) over bundle.corpus_df
# ---------------------------
def dense_retrieve_bge(
    bundle: IRDatasetBundle,
    dataset_name: str,
    query_col: str = "query",
    doc_col: str = "passage",
    qid_col: str = "qid",
    docid_col: str = "docid",
    top_k_dense: int = 100,
    model_name: str = "BAAI/bge-large-en-v1.5",
    corpus_mode: str = "judged",
) -> pd.DataFrame:
    logger.info(f"Loading FlagEmbedding retriever: {model_name}")

    model = get_flag_model(model_name)

    queries_df = bundle.queries_df
    corpus_df = bundle.corpus_df

    # minimal sanity checks (before we touch ordering)
    for col in (qid_col, query_col):
        if col not in queries_df.columns:
            raise ValueError(f"queries_df missing required column '{col}'")
    for col in (docid_col, doc_col):
        if col not in corpus_df.columns:
            raise ValueError(f"corpus_df missing required column '{col}'")

    # ---- deterministic corpus order for FAISS caching ----
    corpus_df = corpus_df.copy()
    corpus_df[docid_col] = corpus_df[docid_col].astype(str)
    corpus_df = corpus_df.sort_values(docid_col).reset_index(drop=True)

    # keep qid stable + preserve complexity if present
    queries_df = queries_df.copy()
    queries_df[qid_col] = queries_df[qid_col].astype(str)

    # Cache paths
    emb_path = _embedding_path(dataset_name, corpus_mode)
    index_path = _faiss_path(dataset_name, corpus_mode)
    docids_path = _docids_path(dataset_name, corpus_mode)

    # Current corpus signature (ORDER matters)
    docids = corpus_df[docid_col].astype(str).tolist()

    def cache_matches_current_order() -> bool:
        if not docids_path.exists():
            return False
        try:
            cached = np.load(docids_path, allow_pickle=True).tolist()
            cached = [str(x) for x in cached]
            return cached == docids
        except Exception:
            return False

    # If cache exists but order mismatches, force rebuild (delete stale files)
    if (emb_path.exists() or index_path.exists()) and not cache_matches_current_order():
        logger.warning(
            f"[dense_retrieve_bge] Cache docid-order mismatch for {dataset_name} ({corpus_mode}). "
            "Deleting stale embeddings/index and rebuilding."
        )
        for p in (emb_path, index_path, docids_path):
            if p.exists():
                p.unlink()

    passages = corpus_df[doc_col].astype(str).tolist()

    # ---- embeddings cache ----
    if emb_path.exists():
        doc_embeddings = np.load(emb_path, mmap_mode="r").astype(np.float32, copy=False)

        norms = np.linalg.norm(doc_embeddings[: min(1000, len(doc_embeddings))], axis=1)
        if not (0.9 <= float(norms.mean()) <= 1.1):
            logger.info("[dense_retrieve_bge] Upgrading cached embeddings to L2-normalized format.")
            doc_embeddings = np.array(doc_embeddings, dtype=np.float32, copy=True)
            faiss.normalize_L2(doc_embeddings)
            np.save(emb_path, doc_embeddings)
            if index_path.exists():
                index_path.unlink()
            doc_embeddings = np.load(emb_path, mmap_mode="r").astype(np.float32, copy=False)


        if doc_embeddings.shape[0] != len(passages):
            logger.warning(
                f"[dense_retrieve_bge] Cached embeddings mismatch "
                f"({doc_embeddings.shape[0]} != {len(passages)}). Re-encoding corpus + rebuilding FAISS."
            )
            if index_path.exists():
                index_path.unlink()
            doc_embeddings = model.encode_corpus(passages).astype(np.float32, copy=False)
            faiss.normalize_L2(doc_embeddings)
            np.save(emb_path, doc_embeddings)
            np.save(docids_path, np.array(docids, dtype=object))

    else:
        doc_embeddings = model.encode_corpus(passages).astype(np.float32, copy=False)
        faiss.normalize_L2(doc_embeddings)
        np.save(emb_path, doc_embeddings)
        np.save(docids_path, np.array(docids, dtype=object))

    # Normalize embeddings for inner product search
    dim = int(doc_embeddings.shape[1])

    # Build or load FAISS index
    index = load_faiss(doc_embeddings, dim, index_path)

    # Query processing
    cols = [qid_col, query_col]
    if "complexity" in queries_df.columns:
        cols.append("complexity")
    unique_queries = queries_df[cols].drop_duplicates()

    results = []
    for _, row in tqdm(unique_queries.iterrows(), total=len(unique_queries)):
        qid = str(row[qid_col])
        query = row[query_col]
        complexity = row["complexity"] if "complexity" in unique_queries.columns else None

        qemb = model.encode_queries([str(query)]).astype(np.float32)
        faiss.normalize_L2(qemb)

        k = min(int(top_k_dense), len(corpus_df))
        D, I = index.search(qemb, k)

        for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
            if idx < 0 or idx >= len(corpus_df):
                continue
            results.append(
                {
                    "qid": qid,
                    "query": query,
                    "complexity": complexity,
                    "docid": corpus_df.iloc[idx][docid_col],
                    "passage": passages[idx],
                    "dense_score": float(score),
                    "dense_rank": rank,
                }
            )

    out_df = pd.DataFrame(results)
    logger.info(f"Dense retrieval completed — {len(out_df)} query-doc pairs.")
    return out_df


# ---------------------------
# Cross-encoder reranking (UNCHANGED logic)
# ---------------------------
def rerank_bge(
    df: pd.DataFrame,
    query_col: str = "query",
    passage_col: str = "passage",
    model_name: str = "BAAI/bge-reranker-large",
    top_k: int = 10,
) -> pd.DataFrame:
    logger.info(f"Loading BGE Reranker: {model_name}")
    reranker = get_flag_reranker(model_name)

    reranked = []
    for qid, group in df.groupby("qid"):
        query = group[query_col].iloc[0]
        passages = group[passage_col].tolist()
        logger.info(f"[rerank_bge] reranking {len(passages)} candidates for qid={qid}")

        pairs = [[query, p] for p in passages]
        scores = []
        bs = 32
        for start in range(0, len(pairs), bs):
            batch = pairs[start : start + bs]
            out = reranker.compute_score(batch)
            if isinstance(out, (list, tuple, np.ndarray)):
                scores.extend([float(x) for x in out])
            else:
                scores.append(float(out))

        if len(scores) != len(passages):
            raise RuntimeError(
                f"[rerank_bge] got {len(scores)} scores for {len(passages)} passages (qid={qid})"
            )

        group = group.copy()
        group["cross_score"] = scores
        group = group.sort_values("cross_score", ascending=False).head(top_k)
        group["rerank"] = range(1, len(group) + 1)
        reranked.append(group)

    reranked_df = pd.concat(reranked, ignore_index=True) if reranked else df.head(0)
    logger.info(f"Reranking complete — {len(reranked_df)} pairs kept (top-{top_k} per query).")
    return reranked_df

def run_bge(
    bundle: IRDatasetBundle,
    dataset_name: str,
    top_k_dense: int,
    top_k: int,
    model_name: str = "BAAI/bge-large-en-v1.5",
    reranker_name: str = "BAAI/bge-reranker-large",
    corpus_mode: str = "judged",
) -> pd.DataFrame:
    bge_results_df = dense_retrieve_bge(
        bundle=bundle,
        dataset_name=dataset_name,
        top_k_dense=top_k_dense,
        model_name=model_name,
        corpus_mode=corpus_mode,
    )
    reranked_df = rerank_bge(bge_results_df, top_k=top_k, model_name=reranker_name)
    logger.info("Pipeline complete — returning final reranked DataFrame.")
    return reranked_df


## STRATEGY 2 entrypoint (bundle-based)
def run_dense_retriever(
    bundle: IRDatasetBundle,
    model_cfg: Dict[str, Any],
    strategy_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    top_k: int,
):
    # ---- derive a stable dataset name WITHOUT touching datasets.yaml ----
    if "year" in dataset_cfg:
        dataset_name = f"trecdl_{dataset_cfg['year']}"
    elif dataset_cfg.get("pyserini_prebuilt_index"):
        # beir + nq (safe because indices differ)
        dataset_name = dataset_cfg["pyserini_prebuilt_index"]
    else:
        raise ValueError(
            "Cannot derive a stable dataset name for dense cache. "
            "Expected 'year' (trecdl) or 'pyserini_prebuilt_index' (beir/nq)."
        )

    corpus_mode = "full" if dataset_cfg.get("full_corpus") else "judged"

    dense_df = run_bge(
        bundle=bundle,
        dataset_name=dataset_name,
        top_k_dense=int(strategy_cfg.get("top_k_dense", 100)),
        top_k=int(top_k),
        model_name=model_cfg.get("retriever", "BAAI/bge-large-en-v1.5"),
        reranker_name=model_cfg.get("reranker", "BAAI/bge-reranker-large"),
        corpus_mode=corpus_mode,
    )
    return dense_df.to_dict(orient="records")
