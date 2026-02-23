# src/evaluation/eval_functional.py
import transformers
transformers.logging.set_verbosity_error()

import argparse
import json
import logging
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from bert_score import score as bert_score

from src.managers.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)

# -------------------------
# Config
# -------------------------
K = 10
REL_THRESHOLD = 1.0  # relevant if rel >= 1 for binary metrics; nDCG uses graded rel.

# Grounding config (UNIFIED): answer nuggets vs evidence nuggets
GROUND_THR = 0.2
MIN_NUGGET_WORDS = 3  # nuggets must have >= 3 words

# Strategy name in JSONL
FUNCTIONAL_STRATEGY_NAME = "functional_agents"


# -------------------------
# IO
# -------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# -------------------------
# Dataset key normalization (GENERAL)
# -------------------------
def normalize_dataset_key(ds: str) -> Optional[str]:
    """
    Normalize dataset key into one of:
      - trecdl2019 / trecdl2020 / trecdl2021
      - beir_scidocs
      - beir_nq

    Accepts metadata.dataset (preferred) or filename fragments.
    """
    if not ds:
        return None

    s = str(ds).strip().lower().replace("-", "_")

    for suf in ["_judged", "_full"]:
        if s.endswith(suf):
            s = s[: -len(suf)]

    # ---- TREC DL ----
    if "trecdl" in s or "trec_dl" in s or "msmarco" in s:
        if "2019" in s:
            return "trecdl2019"
        if "2020" in s:
            return "trecdl2020"
        if "2021" in s:
            return "trecdl2021"

    # ---- BEIR / SCIDOCS ----
    if "scidocs" in s:
        return "beir_scidocs"

    # ---- BEIR / NQ ----
    if "beir_nq" in s or ("natural" in s and "question" in s) or s.endswith("_nq") or s == "nq":
        return "beir_nq"

    return None


def load_bundle_from_dataset_key(dataset_key: str):
    """
    Loads the right IRDatasetBundle (queries + corpus + qrels) given normalized key.
    """
    dm = DatasetManager()

    if dataset_key.startswith("trecdl"):
        year = dataset_key.replace("trecdl", "")  # "2019" / "2020" / "2021"
        bundle, _ = dm.load_bundle(f"trecdl_{year}", limit=None)
        return bundle

    if dataset_key == "beir_scidocs":
        bundle, _ = dm.load_bundle("beir_scidocs", limit=None)
        return bundle

    if dataset_key == "beir_nq":
        bundle, _ = dm.load_bundle("beir_nq", limit=None)
        return bundle

    raise ValueError(f"Unsupported dataset_key: {dataset_key}")


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
    qcol = "qid" if "qid" in queries_df.columns else ("query_id" if "query_id" in queries_df.columns else None)
    if qcol is None:
        logger.warning(f"queries_df missing qid column. Found: {list(queries_df.columns)}")
        return {}
    if "complexity" not in queries_df.columns:
        logger.warning(f"queries_df missing 'complexity' column. Found: {list(queries_df.columns)}")
        return {}

    mp: Dict[str, str] = {}
    for _, row in queries_df.iterrows():
        qid = str(row[qcol])
        comp = row.get("complexity")
        if comp is not None and str(comp).strip():
            mp[qid] = str(comp).strip().lower()
    return mp


def is_rel(rel: float, thr: float = REL_THRESHOLD) -> bool:
    return rel >= thr


# -------------------------
# Retrieval metrics @K
# -------------------------
def accuracy_hit_at_k(ranked: List[str], relmap: Dict[str, float], k: int = K) -> float:
    return 1.0 if any(d in relmap and is_rel(relmap[d]) for d in ranked[:k]) else 0.0


def recall_at_k(ranked: List[str], relmap: Dict[str, float], k: int = K) -> float:
    relevant = [d for d, r in relmap.items() if is_rel(r)]
    if not relevant:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relmap and is_rel(relmap[d]))
    return hits / len(relevant)


def mrr_at_k(ranked: List[str], relmap: Dict[str, float], k: int = K) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if d in relmap and is_rel(relmap[d]):
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: List[str], relmap: Dict[str, float], k: int = K) -> float:
    def dcg(vals: List[float]) -> float:
        s = 0.0
        for i, v in enumerate(vals, start=1):
            s += (2.0 ** v - 1.0) / math.log2(i + 1.0)
        return s

    gains = [float(relmap.get(d, 0.0)) for d in ranked[:k]]
    ideal = sorted([float(r) for r in relmap.values()], reverse=True)[:k]
    denom = dcg(ideal)
    return 0.0 if denom == 0.0 else dcg(gains) / denom


# -------------------------
# Extraction helpers (functional_agents)
# -------------------------
_DOCID_PARENS_RE = re.compile(r"\s*\(docid=[^)]+\)\s*", re.IGNORECASE)
_DOCID_BRACKETS_RE = re.compile(r"\s*\[docid=[^\]]+\]\s*", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def strip_docid_markers(s: str) -> str:
    s = _DOCID_PARENS_RE.sub(" ", s)
    s = _DOCID_BRACKETS_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def extract_qid(rec: Dict[str, Any]) -> Optional[str]:
    v = rec.get("qid") or rec.get("query_id") or rec.get("q_id") or rec.get("id")
    return None if v is None else str(v)


def extract_dataset_key(rec: Dict[str, Any]) -> Optional[str]:
    md = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
    for v in [md.get("dataset"), rec.get("dataset")]:
        if isinstance(v, str) and v.strip():
            return normalize_dataset_key(v)
    return None


def extract_answer_text(rec: Dict[str, Any]) -> str:
    v = rec.get("final_answer") or rec.get("answer") or rec.get("response") or rec.get("generation") or ""
    return v.strip() if isinstance(v, str) else ""


def extract_ranked_docids_topk(rec: Dict[str, Any], k: int = K) -> List[str]:
    """
    Functional records use a single retrieved_docs ranked list.
    We'll score this list directly (stable even if 'subquery' fields are missing).
    """
    docs = rec.get("retrieved_docs")
    if not isinstance(docs, list):
        return []

    items = [d for d in docs if isinstance(d, dict)]
    if any("rank" in d for d in items):
        try:
            items.sort(key=lambda x: int(x.get("rank", 10**9)))
        except Exception:
            pass

    ranked: List[str] = []
    for d in items:
        docid = d.get("docid") or d.get("doc_id") or d.get("docno") or d.get("pid") or d.get("id")
        if docid is None:
            continue
        ranked.append(str(docid))
        if len(ranked) >= k:
            break
    return ranked


def extract_evidence_texts_topk(rec: Dict[str, Any], k: int = K) -> List[str]:
    """
    UNIFIED: evidence comes from retrieved passages (top-k), then we nuggetize them.
    """
    docs = rec.get("retrieved_docs")
    if not isinstance(docs, list):
        return []

    items = [d for d in docs if isinstance(d, dict)]
    if any("rank" in d for d in items):
        try:
            items.sort(key=lambda x: int(x.get("rank", 10**9)))
        except Exception:
            pass

    out: List[str] = []
    for d in items[:k]:
        txt = d.get("passage") or d.get("text") or d.get("contents") or ""
        if isinstance(txt, str) and txt.strip():
            out.append(strip_docid_markers(txt.strip()))
    return out


# -------------------------
# Nuggets (>=3 words) + ROUGE-L grounding
# -------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_PUNCT_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def normalize_nugget(s: str) -> str:
    s = s.lower().strip()
    s = _WS_RE.sub(" ", s)
    return _PUNCT_EDGE_RE.sub("", s)


def nugget_extractor_list(text: str) -> List[str]:
    """
    UNIFIED: returns LIST of normalized nuggets (deduped), each with >= MIN_NUGGET_WORDS.
    """
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    nuggets: List[str] = []
    seen = set()
    for sent in parts:
        n = normalize_nugget(sent)
        if len(n.split()) >= MIN_NUGGET_WORDS and n not in seen:
            nuggets.append(n)
            seen.add(n)
    return nuggets


def lcs_len(a: List[str], b: List[str]) -> int:
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = prev + 1 if a[i - 1] == b[j - 1] else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l_f1(candidate: str, reference: str) -> float:
    c = _TOKEN_RE.findall(candidate.lower())
    r = _TOKEN_RE.findall(reference.lower())
    if not c or not r:
        return 0.0
    lcs = lcs_len(c, r)
    p, q = lcs / len(c), lcs / len(r)
    return 0.0 if p + q == 0 else 2 * p * q / (p + q)


def grounding_coverage_rouge(
    answer: str,
    evidence_texts: List[str],
    *,
    thr: float = GROUND_THR,
) -> Tuple[Optional[float], int, int]:
    """
    UNIFIED: answer nuggets vs evidence nuggets (ROUGE-L >= thr).

    Returns:
      coverage (0..1 or None), grounded_count, total_answer_nuggets
    """
    ans = nugget_extractor_list(answer)
    if not ans:
        return None, 0, 0

    # build evidence nugget set
    ev_set = set()
    for t in (evidence_texts or []):
        if isinstance(t, str) and t.strip():
            ev_set.update(nugget_extractor_list(t))

    if not ev_set:
        return None, 0, len(ans)

    ev_list = list(ev_set)

    grounded = 0
    for a in ans:
        best = 0.0
        for e in ev_list:
            best = max(best, rouge_l_f1(a, e))
            if best >= thr:
                break
        if best >= thr:
            grounded += 1

    return grounded / len(ans), grounded, len(ans)


def gold_fact_recall_retrieved_exact(
    answer: str,
    evidence_texts: List[str],
) -> Tuple[Optional[float], int, int]:
    """
    UNIFIED: "GoldFactRecall_retrieved" as exact nugget overlap:
      |A ∩ E| / |E|
    where:
      A = answer nugget set
      E = evidence nugget set (from retrieved passages)
    """
    if not answer:
        return None, 0, 0

    A = set(nugget_extractor_list(answer))
    if not A:
        return None, 0, 0

    E = set()
    for t in (evidence_texts or []):
        if isinstance(t, str) and t.strip():
            E.update(nugget_extractor_list(t))

    if not E:
        return None, 0, len(A)

    return (len(A & E) / len(E), len(E), len(A))


# -------------------------
# BERTScore (minimal)
# -------------------------
def batched_bertscore_f1(pairs: List[Tuple[str, str]], batch_size: int = 64) -> List[float]:
    out: List[float] = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        cands = [c for c, _ in chunk]
        refs = [r for _, r in chunk]
        _, _, F1 = bert_score(
            cands,
            refs,
            lang="en",
            model_type="roberta-base",
            verbose=False,
            batch_size=batch_size,
        )
        out.extend([float(x) for x in F1])
    return out


# -------------------------
# Evaluation (functional_agents) + complexity
# -------------------------
def evaluate_functional(
    *,
    model: str,
    dataset: str,
    records: List[Dict[str, Any]],
    qrels_map: Dict[str, Dict[str, float]],
    complexity_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    per_q_rows: List[Dict[str, Any]] = []
    complexity_map = complexity_map or {}

    # For per-qid mean BERTScore across evidence docs
    bert_pairs: List[Tuple[str, str]] = []
    bert_counts: Dict[str, int] = {}
    bert_order: List[str] = []

    for rec in records:
        qid = extract_qid(rec)
        if not qid:
            continue

        comp = complexity_map.get(qid, "unknown")

        # retrieval
        ranked_docids = extract_ranked_docids_topk(rec, k=K)
        relmap = qrels_map.get(qid, {})

        if not ranked_docids or not relmap:
            acc10 = rec10 = mrr10 = ndcg10 = 0.0
        else:
            acc10 = accuracy_hit_at_k(ranked_docids, relmap, k=K)
            rec10 = recall_at_k(ranked_docids, relmap, k=K)
            mrr10 = mrr_at_k(ranked_docids, relmap, k=K)
            ndcg10 = ndcg_at_k(ranked_docids, relmap, k=K)

        # generation + evidence (UNIFIED: retrieved passages)
        answer = strip_docid_markers(extract_answer_text(rec))
        evidence_texts = extract_evidence_texts_topk(rec, k=K)

        # grounding: answer nuggets vs evidence nuggets
        info_coverage = None
        answer_fact_precision_retrieved = None
        gold_fact_recall_retrieved = None
        rougeL = None

        n_grounded = None
        n_answer_nuggets = None
        n_evidence_nuggets = None

        if answer and evidence_texts:
            cov, grounded_cnt, ans_cnt = grounding_coverage_rouge(
                answer,
                evidence_texts,
                thr=GROUND_THR,
            )
            info_coverage = cov
            answer_fact_precision_retrieved = cov
            n_grounded = grounded_cnt
            n_answer_nuggets = ans_cnt

            gfr, nE, nA = gold_fact_recall_retrieved_exact(answer, evidence_texts)
            gold_fact_recall_retrieved = gfr
            n_evidence_nuggets = nE
            if n_answer_nuggets is None:
                n_answer_nuggets = nA

            # quality ROUGE-L: mean(answer vs each retrieved passage)
            rougeL = sum(rouge_l_f1(answer, t) for t in evidence_texts) / len(evidence_texts)

            # quality BERTScore: mean(answer vs each retrieved passage)
            bert_order.append(qid)
            bert_counts[qid] = bert_counts.get(qid, 0) + len(evidence_texts)
            for t in evidence_texts:
                bert_pairs.append((answer, t))

        per_q_rows.append(
            {
                "model": model,
                "strategy": FUNCTIONAL_STRATEGY_NAME,
                "dataset": dataset,
                "complexity": comp,
                "qid": qid,
                # retrieval
                "accuracy@10": acc10,
                "recall@10": rec10,
                "mrr@10": mrr10,
                "ndcg@10": ndcg10,
                "n_relevant": sum(1 for _, r in relmap.items() if is_rel(r)) if relmap else 0,
                # generation / evidence (UNIFIED naming)
                "info_coverage": info_coverage,
                "gold_fact_recall_retrieved": gold_fact_recall_retrieved,
                "answer_fact_precision_retrieved": answer_fact_precision_retrieved,
                "quality_rougeL": rougeL,
                "quality_bertscore": None,
                # counts
                "n_evidence_nuggets": n_evidence_nuggets,
                "n_answer_nuggets": n_answer_nuggets,
                "n_grounded_answer_nuggets": n_grounded,
            }
        )

    # Fill BERTScore means per qid
    if bert_pairs:
        f1s = batched_bertscore_f1(bert_pairs, batch_size=64)
        qid_to_mean: Dict[str, float] = {}
        idx = 0
        for qid in bert_order:
            n = bert_counts.get(qid, 0)
            if n <= 0:
                continue
            qid_to_mean[qid] = float(sum(f1s[idx : idx + n]) / n)
            idx += n

        for row in per_q_rows:
            qid = row["qid"]
            if qid in qid_to_mean:
                row["quality_bertscore"] = qid_to_mean[qid]

    df = pd.DataFrame(per_q_rows)

    summary = dict(
        model=model,
        strategy=FUNCTIONAL_STRATEGY_NAME,
        dataset=dataset,
        n_queries=int(df["qid"].nunique()) if not df.empty else 0,
    )

    if not df.empty:
        for col in [
            "accuracy@10",
            "recall@10",
            "mrr@10",
            "ndcg@10",
            "info_coverage",
            "gold_fact_recall_retrieved",
            "answer_fact_precision_retrieved",
            "quality_rougeL",
            "quality_bertscore",
        ]:
            summary[col] = float(df[col].dropna().mean()) if col in df.columns else None

        summary["avg_n_evidence_nuggets"] = float(df["n_evidence_nuggets"].dropna().mean()) if "n_evidence_nuggets" in df.columns else None
        summary["avg_n_answer_nuggets"] = float(df["n_answer_nuggets"].dropna().mean()) if "n_answer_nuggets" in df.columns else None

    return df, summary


def summarize_by_complexity(per_q_df: pd.DataFrame) -> pd.DataFrame:
    if per_q_df.empty or "complexity" not in per_q_df.columns:
        return pd.DataFrame()

    return (
        per_q_df.groupby(["model", "dataset", "complexity"], dropna=False)
        .agg(
            n_queries=("qid", "nunique"),
            accuracy_at_10=("accuracy@10", "mean"),
            recall_at_10=("recall@10", "mean"),
            mrr_at_10=("mrr@10", "mean"),
            ndcg_at_10=("ndcg@10", "mean"),
            info_coverage=("info_coverage", "mean"),
            gold_fact_recall_retrieved=("gold_fact_recall_retrieved", "mean"),
            answer_fact_precision_retrieved=("answer_fact_precision_retrieved", "mean"),
            quality_rougeL=("quality_rougeL", "mean"),
            quality_bertscore=("quality_bertscore", "mean"),
            avg_n_evidence_nuggets=("n_evidence_nuggets", "mean"),
            avg_n_answer_nuggets=("n_answer_nuggets", "mean"),
        )
        .reset_index()
    )


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate functional_agents on any dataset (retrieval@10 + UNIFIED nugget grounding/quality vs retrieved evidence)"
    )
    parser.add_argument("--inputs", required=True, help="Directory containing JSONL result files")
    parser.add_argument("--out", required=True, help="Output CSV file for summary metrics")
    parser.add_argument("--detailed-out", help="Optional output CSV for per-query metrics")
    parser.add_argument("--by-complexity-out", help="Optional output CSV for complexity-wise metrics")
    parser.add_argument(
        "--years",
        default="2019,2020,2021",
        help="TREC DL years to include (only applies to trecdl), e.g., 2020 or 2019,2020",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    years = [y.strip() for y in args.years.split(",") if y.strip()]
    allowed_trecdl = {f"trecdl{y}" for y in years}

    input_dir = Path(args.inputs)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    t0 = time.perf_counter()

    # 1) Collect relevant files: must contain 'functional' in filename
    files = []
    for f in input_dir.glob("*.jsonl"):
        name = f.name.lower()
        if "functional_agents" in name or "functional" in name:
            files.append(f)

    if not files:
        raise RuntimeError("No files matched: '*.jsonl' containing 'functional' in the filename.")

    # 2) Group by (model, dataset_key)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for f in files:
        rows = load_jsonl(f)

        # Determine model name cheaply from filename prefix (everything before '__')
        stem = f.stem
        model = stem.split("__")[0] if "__" in stem else stem

        for r in rows:
            strat = r.get("strategy")
            if strat and strat != FUNCTIONAL_STRATEGY_NAME:
                continue

            ds_key = extract_dataset_key(r)
            if ds_key is None:
                ds_key = normalize_dataset_key(f.name)
            if ds_key is None:
                continue

            if ds_key.startswith("trecdl") and ds_key not in allowed_trecdl:
                continue

            grouped[(model, ds_key)].append(r)

    if not grouped:
        raise RuntimeError("No functional_agents records found after filtering by strategy/years.")

    # 3) Cache qrels + complexity per dataset_key
    qrels_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
    complexity_cache: Dict[str, Dict[str, str]] = {}

    summaries: List[Dict[str, Any]] = []
    detailed_frames: List[pd.DataFrame] = []

    for (model, ds_key), records in grouped.items():
        logger.info(f"Evaluating: model={model} dataset={ds_key} n_rows={len(records)}")

        if ds_key not in qrels_cache:
            bundle = load_bundle_from_dataset_key(ds_key)
            qrels_cache[ds_key] = make_qrels_map(bundle.qrels_df)
            complexity_cache[ds_key] = make_complexity_map(bundle.queries_df)

        per_q_df, summ = evaluate_functional(
            model=model,
            dataset=ds_key,
            records=records,
            qrels_map=qrels_cache[ds_key],
            complexity_map=complexity_cache.get(ds_key, {}),
        )
        summaries.append(summ)
        if not per_q_df.empty:
            detailed_frames.append(per_q_df)

    # 4) Save outputs
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_path, index=False)
    logger.info(f"Saved summary to {out_path}")

    detailed_df = pd.concat(detailed_frames, ignore_index=True) if detailed_frames else pd.DataFrame()

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

    print("\nFunctional Agents Evaluation Summary (@10 + UNIFIED nugget grounding/quality):")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    if args.by_complexity_out and not detailed_df.empty:
        byc_df = summarize_by_complexity(detailed_df)
        if not byc_df.empty:
            print("\nFunctional Agents Evaluation By Complexity (@10 + UNIFIED nugget grounding/quality):")
            print("=" * 100)
            print(byc_df.to_string(index=False))

    logger.info(f"Total eval time: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
