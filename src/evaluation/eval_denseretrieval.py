# src/evaluation/eval_denseretrieval.py
import argparse
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.managers.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)

K = 10
REL_THRESHOLD = 1.0  # relevant if rel >= 1 for binary metrics; nDCG uses graded rel.

# Matches your filenames like:
# none__denseretrieval__nq__abcd.jsonl
# claude-4-5-haiku__denseretrieval__beir__abcd.jsonl
FILENAME_RE = re.compile(
    r"^(?P<model>.+?)__(?P<strategy>rag|human_proxy_rag|denseretrieval|hierarchy_agents|peer_agents|functional_agents)__(?P<dataset>.+?)__.*\.jsonl$"
)
# -------------------------
# IO
# -------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_filename_meta(name: str) -> Dict[str, str]:
    m = FILENAME_RE.match(name)
    if not m:
        return {}
    return {"model": m.group("model"), "strategy": m.group("strategy"), "dataset": m.group("dataset")}


# -------------------------
# Dataset key normalization
# -------------------------
def normalize_dataset_key(ds: str) -> Optional[str]:
    """
    Your ecosystem uses a few variants:
      - trecdl2020 (filename)
      - trecdl_2020 (some metadata)
      - trecdl_2020_judged (some agent metadata)
      - beir
      - nq
    Normalize to one of:
      {beir, nq, trecdl2019, trecdl2020, trecdl2021}
    """
    if not ds:
        return None
    s = str(ds).strip().lower()
    s = s.replace("-", "_")

    # strip suffixes
    for suf in ["_judged", "_full"]:
        if s.endswith(suf):
            s = s[: -len(suf)]

    # trecdl_2020 -> trecdl2020
    if s.startswith("trecdl_"):
        s = s.replace("trecdl_", "trecdl")
    if s.startswith("trecdl") and len(s) == len("trecdl") + 4 and s[-4:].isdigit():
        return s

    if s in {"beir", "scidocs", "beir_scidocs"}:
        return "beir"
    if s in {"nq", "natural_questions", "naturalquestion"}:
        return "nq"

    # some agent outputs may store "trecdl_2020" under metadata.dataset as "trecdl_2020"
    if s.startswith("trecdl") and any(y in s for y in ["2019", "2020", "2021"]):
        # keep only digits at end
        m = re.search(r"(2019|2020|2021)", s)
        if m:
            return f"trecdl{m.group(1)}"

    return None


# -------------------------
# Qrels mapping + Complexity mapping
# -------------------------
def make_qrels_map(qrels_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    qcol = "qid" if "qid" in qrels_df.columns else ("query_id" if "query_id" in qrels_df.columns else None)
    dcol = "docid" if "docid" in qrels_df.columns else ("doc_id" if "doc_id" in qrels_df.columns else None)
    rcol = next((c for c in ["relevance", "rel", "score", "label", "judgment"] if c in qrels_df.columns), None)

    if qcol is None or dcol is None or rcol is None:
        raise ValueError(f"qrels_df needs qid/docid/relevance-ish columns. Found: {list(qrels_df.columns)}")

    mp: Dict[str, Dict[str, float]] = {}
    for _, row in qrels_df.iterrows():
        qid = str(row[qcol])
        docid = str(row[dcol])
        rel = float(row[rcol])
        mp.setdefault(qid, {})[docid] = rel
    return mp


def make_complexity_map(queries_df: pd.DataFrame) -> Dict[str, str]:
    """
    Build qid -> complexity label map (for complexity-wise reporting).

    Tries these columns:
      qid: "qid" or "query_id"
      complexity: "complexity", "complexity_label", "difficulty", "label"
    """
    qcol = "qid" if "qid" in queries_df.columns else ("query_id" if "query_id" in queries_df.columns else None)
    ccol = next((c for c in ["complexity", "complexity_label", "difficulty", "label"] if c in queries_df.columns), None)

    if qcol is None or ccol is None:
        logger.warning(
            f"queries_df missing qid/complexity columns; complexity-wise metrics will be 'unknown'. "
            f"Found: {list(queries_df.columns)}"
        )
        return {}

    mp: Dict[str, str] = {}
    for _, row in queries_df.iterrows():
        qid = str(row[qcol])
        comp = row.get(ccol)
        comp_s = "" if comp is None else str(comp).strip()
        if comp_s:
            mp[qid] = comp_s
    return mp


def is_rel(rel: float, thr: float = REL_THRESHOLD) -> bool:
    return rel >= thr


# -------------------------
# Metrics @10
# -------------------------
def accuracy_hit_at_10(ranked: List[str], relmap: Dict[str, float]) -> float:
    return 1.0 if any(d in relmap and is_rel(relmap[d]) for d in ranked[:K]) else 0.0


def recall_at_10(ranked: List[str], relmap: Dict[str, float]) -> float:
    relevant = [d for d, r in relmap.items() if is_rel(r)]
    if not relevant:
        return 0.0
    hits = sum(1 for d in ranked[:K] if d in relmap and is_rel(relmap[d]))
    return hits / len(relevant)


def mrr_at_10(ranked: List[str], relmap: Dict[str, float]) -> float:
    for i, d in enumerate(ranked[:K], start=1):
        if d in relmap and is_rel(relmap[d]):
            return 1.0 / i
    return 0.0


def ndcg_at_10(ranked: List[str], relmap: Dict[str, float]) -> float:
    def dcg(vals: List[float]) -> float:
        s = 0.0
        for i, v in enumerate(vals, start=1):
            s += (2.0 ** v - 1.0) / math.log2(i + 1.0)
        return s

    gains = [float(relmap.get(d, 0.0)) for d in ranked[:K]]
    ideal = sorted([float(r) for r in relmap.values()], reverse=True)[:K]
    denom = dcg(ideal)
    return 0.0 if denom == 0.0 else dcg(gains) / denom


# -------------------------
# Dataset loading
# -------------------------

def load_bundle(dataset_key: str):
    dm = DatasetManager()

    # trecdl2019 -> trecdl_2019
    if dataset_key.startswith("trecdl"):
        year = dataset_key.replace("trecdl", "")
        ds_name = f"trecdl_{year}"  # matches your config naming
        bundle, _ = dm.load_bundle(ds_name, limit=None)
        return bundle

    if dataset_key == "beir":
        bundle, _ = dm.load_bundle("beir", limit=None)
        return bundle

    if dataset_key == "nq":
        bundle, _ = dm.load_bundle("nq", limit=None)
        return bundle

    raise ValueError(f"Unsupported dataset key: {dataset_key}")

# -------------------------
# Dense result parsing (your schema)
# -------------------------
def extract_qid(rec: Dict[str, Any]) -> Optional[str]:
    v = rec.get("qid") or rec.get("query_id") or rec.get("q_id") or rec.get("id")
    return None if v is None else str(v)


def extract_docid(rec: Dict[str, Any]) -> Optional[str]:
    v = rec.get("docid") or rec.get("doc_id") or rec.get("docno") or rec.get("pid")
    return None if v is None else str(v)


def extract_final_rank(rec: Dict[str, Any]) -> Optional[int]:
    """
    Dense retrieval output:
      - rerank: final rank after reranker (preferred)
      - dense_rank: rank from initial dense retrieval (fallback)
    """
    if rec.get("rerank") is not None:
        try:
            return int(rec["rerank"])
        except Exception:
            return None
    if rec.get("dense_rank") is not None:
        try:
            return int(rec["dense_rank"])
        except Exception:
            return None
    # fallback: sometimes uses "rank"
    if rec.get("rank") is not None:
        try:
            return int(rec["rank"])
        except Exception:
            return None
    return None


# -------------------------
# Evaluation
# -------------------------
def evaluate_group(
    *,
    model: str,
    dataset: str,
    records: List[Dict[str, Any]],
    qrels_map: Dict[str, Dict[str, float]],
    complexity_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Builds per-query ranked lists from row-per-doc format, computes metrics, and attaches complexity.
    """
    complexity_map = complexity_map or {}

    # Build per-query ranked lists from row-per-doc format
    per_q_docs: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    for rec in records:
        qid = extract_qid(rec)
        did = extract_docid(rec)
        rnk = extract_final_rank(rec)

        if not qid or not did or rnk is None:
            continue

        per_q_docs[qid].append((rnk, did))

    per_q_rows: List[Dict[str, Any]] = []

    for qid, pairs in per_q_docs.items():
        pairs.sort(key=lambda t: t[0])
        ranked = [d for _, d in pairs]
        relmap = qrels_map.get(qid, {})
        comp = complexity_map.get(qid, "unknown")

        row = {
            "model": model,
            "strategy": "denseretrieval",
            "dataset": dataset,
            "complexity": comp,
            "qid": qid,
            "accuracy@10": accuracy_hit_at_10(ranked, relmap),
            "recall@10": recall_at_10(ranked, relmap),
            "mrr@10": mrr_at_10(ranked, relmap),
            "ndcg@10": ndcg_at_10(ranked, relmap),
            "n_candidates": len(ranked),
            "n_relevant": sum(1 for _, r in relmap.items() if is_rel(r)),
            "top10_docids": " ".join(ranked[:K]),
        }
        per_q_rows.append(row)

    per_q_df = pd.DataFrame(per_q_rows)

    summary = {
        "model": model,
        "strategy": "denseretrieval",
        "dataset": dataset,
        "n_queries": int(per_q_df["qid"].nunique()) if not per_q_df.empty else 0,
    }

    if not per_q_df.empty:
        summary["accuracy@10"] = float(per_q_df["accuracy@10"].mean())
        summary["recall@10"] = float(per_q_df["recall@10"].mean())
        summary["mrr@10"] = float(per_q_df["mrr@10"].mean())
        summary["ndcg@10"] = float(per_q_df["ndcg@10"].mean())

    return per_q_df, summary


def summarize_by_complexity(per_q_df: pd.DataFrame) -> pd.DataFrame:
    """
    Complexity-wise mean metrics, grouped by (model, dataset, complexity).
    """
    if per_q_df.empty or "complexity" not in per_q_df.columns:
        return pd.DataFrame()

    agg = (
        per_q_df.groupby(["model", "dataset", "complexity"], dropna=False)
        .agg(
            n_queries=("qid", "nunique"),
            accuracy_at_10=("accuracy@10", "mean"),
            recall_at_10=("recall@10", "mean"),
            mrr_at_10=("mrr@10", "mean"),
            ndcg_at_10=("ndcg@10", "mean"),
        )
        .reset_index()
    )
    return agg


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dense Retrieval metrics (Accuracy/Recall/MRR/nDCG @10)")
    parser.add_argument("--inputs", type=str, required=True, help="Directory containing JSONL result files")
    parser.add_argument("--out", type=str, required=True, help="Output CSV file for summary metrics")
    parser.add_argument("--detailed-out", type=str, help="Optional output CSV for per-query metrics")
    parser.add_argument("--by-complexity-out", type=str, help="Optional output CSV for complexity-wise metrics")
    parser.add_argument(
        "--datasets",
        default="beir,nq,trecdl2019,trecdl2020,trecdl2021",
        help="Comma-separated allowed datasets (after normalization)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    allowed = {normalize_dataset_key(d.strip()) for d in args.datasets.split(",") if d.strip()}
    allowed = {d for d in allowed if d is not None}

    input_dir = Path(args.inputs)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    # 1) Load and group only Dense Retriever results
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        meta = parse_filename_meta(jsonl_file.name)

        # Prefer filename strategy filter when available
        if meta and meta.get("strategy") != "denseretrieval":
            continue

        try:
            rows = load_jsonl(jsonl_file)
        except Exception as e:
            logger.error(f"Error loading {jsonl_file}: {e}")
            continue

        file_model = meta.get("model") if meta else None
        file_dataset = meta.get("dataset") if meta else None

        for r in rows:
            # Hard filter to avoid hierarchy_agents rows accidentally included in mixed files
            if r.get("strategy") and r.get("strategy") != "denseretrieval":
                continue

            # Dense rows have rerank/dense_rank; if missing, skip
            if r.get("rerank") is None and r.get("dense_rank") is None and r.get("rank") is None:
                continue

            model = file_model or r.get("model", "unknown")
            ds_raw = file_dataset or r.get("dataset") or (r.get("metadata", {}) or {}).get("dataset")
            dataset = normalize_dataset_key(ds_raw)

            if dataset is None or dataset not in allowed:
                continue

            grouped[(model, dataset)].append(r)

    if not grouped:
        logger.error("No Dense Retrieval results found in inputs.")
        return

    # 2) Cache qrels + complexity per dataset
    qrels_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
    complexity_cache: Dict[str, Dict[str, str]] = {}

    summary_rows: List[Dict[str, Any]] = []
    detailed_frames: List[pd.DataFrame] = []

    for (model, dataset), records in grouped.items():
        logger.info(f"Evaluating Dense: model={model} dataset={dataset} n_rows={len(records)}")

        if dataset not in qrels_cache:
            bundle = load_bundle(dataset)
            qrels_cache[dataset] = make_qrels_map(bundle.qrels_df)
            complexity_cache[dataset] = make_complexity_map(bundle.queries_df)

        qrels_map = qrels_cache[dataset]
        comp_map = complexity_cache.get(dataset, {})

        per_q_df, summary = evaluate_group(
            model=model,
            dataset=dataset,
            records=records,
            qrels_map=qrels_map,
            complexity_map=comp_map,
        )

        if "n_queries" in summary:
            logger.info(
                f"  n_queries={summary['n_queries']}  acc@10={summary.get('accuracy@10', 0):.3f}  mrr@10={summary.get('mrr@10', 0):.3f}"
            )

        summary_rows.append(summary)
        if not per_q_df.empty:
            detailed_frames.append(per_q_df)

    # 3) Save outputs
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_path, index=False)
    logger.info(f"Saved summary to {out_path}")

    detailed_df = pd.DataFrame()
    if detailed_frames:
        detailed_df = pd.concat(detailed_frames, ignore_index=True)

    if args.detailed_out:
        detailed_path = Path(args.detailed_out)
        detailed_path.parent.mkdir(parents=True, exist_ok=True)
        detailed_df.to_csv(detailed_path, index=False)
        logger.info(f"Saved detailed to {detailed_path}")

    if args.by_complexity_out:
        byc_df = summarize_by_complexity(detailed_df)
        byc_path = Path(args.by_complexity_out)
        byc_path.parent.mkdir(parents=True, exist_ok=True)
        byc_df.to_csv(byc_path, index=False)
        logger.info(f"Saved by-complexity to {byc_path}")

    # Print summary
    print("\nDense Retrieval Evaluation Summary (@10):")
    print("=" * 70)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No results to display")

    if args.by_complexity_out and not detailed_df.empty:
        byc_df = summarize_by_complexity(detailed_df)
        if not byc_df.empty:
            print("\nDense Retrieval Evaluation By Complexity (@10):")
            print("=" * 70)
            print(byc_df.to_string(index=False))


if __name__ == "__main__":
    main()
