# src/evaluation/eval_human_proxy_rag.py
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

# Shared retrieval/evidence window
K = 10
REF_K = 10  # score quality vs all 10 retrieved docs

# Unified binary relevance threshold for hit/recall/mrr across datasets
REL_THR: Dict[str, float] = {
    "beir": 1.0,
    "nq": 1.0,
    "trecdl2019": 1.0,
    "trecdl2020": 1.0,
    "trecdl2021": 1.0,
}

# -------------------------
# HUMAN_PROXY_RAG-ONLY filename matcher
# -------------------------
FILENAME_RE = re.compile(r"^(?P<model>.+?)__human_proxy_rag__(?P<dataset>.+?)__.*\.jsonl$")

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
    return {"model": m.group("model"), "strategy": "human_proxy_rag", "dataset": m.group("dataset")}


# -------------------------
# Dataset key normalization
# -------------------------
def normalize_dataset_key(ds: str) -> Optional[str]:
    """
    Normalize to one of:
      {beir, nq, trecdl2019, trecdl2020, trecdl2021}
    """
    if not ds:
        return None
    s = str(ds).strip().lower()
    s = s.replace("-", "_")

    for suf in ["_judged", "_full"]:
        if s.endswith(suf):
            s = s[: -len(suf)]

    if s.startswith("trecdl_"):
        s = s.replace("trecdl_", "trecdl")

    if s in {"beir", "scidocs", "beir_scidocs"}:
        return "beir"
    if s in {"nq", "natural_questions", "naturalquestion"}:
        return "nq"

    if s.startswith("trecdl") and any(y in s for y in ["2019", "2020", "2021"]):
        m = re.search(r"(2019|2020|2021)", s)
        if m:
            return f"trecdl{m.group(1)}"

    return None


# -------------------------
# Dataset loading (qrels only needed for retrieval metrics)
# -------------------------
def load_bundle(dataset_key: str):
    dm = DatasetManager()

    if dataset_key.startswith("trecdl"):
        year = int(dataset_key.replace("trecdl", ""))  # trecdl2021 -> 2021
        bundle, _ = dm.load_bundle(f"trecdl_{year}", cfg={}, limit=None)
        return bundle

    if dataset_key == "beir":
        bundle, _ = dm.load_bundle("beir", cfg={}, limit=None)
        return bundle

    if dataset_key == "nq":
        bundle, _ = dm.load_bundle("nq", cfg={}, limit=None)
        return bundle

    raise ValueError(f"Unsupported dataset key: {dataset_key}")


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


def is_rel(rel: float, thr: float) -> bool:
    return rel >= thr


# -------------------------
# Retrieval metrics @10
# -------------------------
def accuracy_hit_at_10(ranked: List[str], relmap: Dict[str, float], thr: float) -> float:
    return 1.0 if any(d in relmap and is_rel(relmap[d], thr) for d in ranked[:K]) else 0.0


def recall_at_10(ranked: List[str], relmap: Dict[str, float], thr: float) -> float:
    relevant = [d for d, r in relmap.items() if is_rel(r, thr)]
    if not relevant:
        return 0.0
    hits = sum(1 for d in ranked[:K] if d in relmap and is_rel(relmap[d], thr))
    return hits / len(relevant)


def mrr_at_10(ranked: List[str], relmap: Dict[str, float], thr: float) -> float:
    for i, d in enumerate(ranked[:K], start=1):
        if d in relmap and is_rel(relmap[d], thr):
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
# Record parsing
# -------------------------
def extract_qid(rec: Dict[str, Any]) -> Optional[str]:
    v = rec.get("qid") or rec.get("query_id") or rec.get("q_id") or rec.get("id")
    return None if v is None else str(v)


def extract_persona(rec: Dict[str, Any]) -> str:
    v = rec.get("persona") or rec.get("profile") or rec.get("assessor") or "unknown"
    return str(v)


def extract_answer_text(rec: Dict[str, Any]) -> str:
    for k in ["answer", "final_answer", "response", "generation", "generated_answer", "output"]:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def extract_ranked_docids(rec: Dict[str, Any]) -> List[str]:
    docs = rec.get("retrieved_docs")
    if isinstance(docs, list) and docs:
        ranked: List[Tuple[int, str]] = []
        for i, item in enumerate(docs):
            if not isinstance(item, dict):
                continue
            did = item.get("docid") or item.get("doc_id") or item.get("id") or item.get("docno")
            if did is None:
                continue
            rank_val = item.get("rank")
            try:
                r = int(rank_val) if rank_val is not None else (i + 1)
            except Exception:
                r = i + 1
            ranked.append((r, str(did)))
        ranked.sort(key=lambda t: t[0])
        return [d for _, d in ranked]
    return []


def extract_retrieved_texts(rec: Dict[str, Any], k: int = K) -> List[str]:
    docs = rec.get("retrieved_docs")
    if not isinstance(docs, list) or not docs:
        return []

    ranked: List[Tuple[int, str]] = []
    for i, item in enumerate(docs):
        if not isinstance(item, dict):
            continue

        rank_val = item.get("rank")
        try:
            rank = int(rank_val) if rank_val is not None else i + 1
        except Exception:
            rank = i + 1

        txt = (
            item.get("passage")
            or item.get("text")
            or item.get("contents")
            or item.get("body")
            or item.get("document")
            or ""
        )
        txt = "" if txt is None else str(txt)
        ranked.append((rank, txt))

    ranked.sort(key=lambda t: t[0])
    return [t for _, t in ranked[:k] if t.strip()]


def docid_signature(rec: Dict[str, Any], k: int = K) -> Tuple[str, ...]:
    return tuple(extract_ranked_docids(rec)[:k])


# -------------------------
# Nuggets (sentence-level placeholder)
# -------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WS_RE = re.compile(r"\s+")
_PUNCT_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$")


def normalize_nugget(s: str) -> str:
    s = s.strip().lower()
    s = _WS_RE.sub(" ", s)
    s = _PUNCT_EDGE_RE.sub("", s)
    return s


def nugget_extractor(text: str, *, min_words: int = 3) -> List[str]:
    if not text:
        return []

    parts = _SENT_SPLIT_RE.split(text)
    nuggets: List[str] = []
    seen = set()

    for sent in parts:
        n = normalize_nugget(sent)
        if not n:
            continue
        if len(n.split()) < min_words:
            continue
        if n not in seen:
            nuggets.append(n)
            seen.add(n)

    return nuggets

def gold_fact_recall_retrieved(
    answer: str,
    retrieved_texts: List[str],
    *,
    k: int = REF_K,
) -> Tuple[Optional[float], int, int]:

    if not answer:
        return None, 0, 0

    gold_nuggets = set().union(*(nugget_extractor(t) for t in (retrieved_texts or [])[:k]))
    answer_nuggets = set(nugget_extractor(answer))

    if not gold_nuggets:
        return None, 0, len(answer_nuggets)

    score = len(answer_nuggets & gold_nuggets) / len(gold_nuggets)
    return score, len(gold_nuggets), len(answer_nuggets)



# -------------------------
# ROUGE-L
# -------------------------
_TOKEN_RE_SIMPLE = re.compile(r"[A-Za-z0-9]+")


def lcs_len(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[m]


def rouge_l_f1(candidate: str, reference: str) -> float:
    c = _TOKEN_RE_SIMPLE.findall(candidate.lower())
    r = _TOKEN_RE_SIMPLE.findall(reference.lower())
    if not c or not r:
        return 0.0
    lcs = lcs_len(c, r)
    prec = lcs / len(c)
    rec = lcs / len(r)
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def grounding_coverage_rouge(
    answer: str,
    evidence_texts: List[str],
    *,
    thr: float = 0.2,
) -> Tuple[Optional[float], int, int]:
    """
    Coverage = grounded_answer_nuggets / answer_nuggets

    Grounding rule:
      an answer nugget is grounded if max_{evidence_nugget} ROUGE-L(answer_nugget, evidence_nugget) >= thr

    IMPORTANT:
      evidence is nuggetized (not whole documents) — matches eval_rag.py.
    """
    ans_nuggets = nugget_extractor(answer)  # >=3 words
    if not ans_nuggets:
        return None, 0, 0

    ev_nuggets = set().union(
        *(nugget_extractor(t) for t in (evidence_texts or []) if isinstance(t, str) and t.strip())
    )
    if not ev_nuggets:
        return None, 0, len(ans_nuggets)

    grounded = 0
    ev_list = list(ev_nuggets)

    for a in ans_nuggets:
        best = 0.0
        for e in ev_list:
            best = max(best, rouge_l_f1(a, e))
            if best >= thr:
                break
        if best >= thr:
            grounded += 1

    return grounded / len(ans_nuggets), grounded, len(ans_nuggets)



# -------------------------
# BERTScore batching
# -------------------------
def _batched_mean_bertscore_f1(
    items: List[Tuple[str, List[str]]],
    *,
    model_type: str = "roberta-base",
    batch_size: int = 64,
    max_pairs_per_call: int = 3000,
) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(items)

    flat_cands: List[str] = []
    flat_refs: List[str] = []
    flat_item_idx: List[int] = []

    for i, (cand, refs) in enumerate(items):
        cand = (cand or "").strip()
        refs = [r for r in (refs or []) if isinstance(r, str) and r.strip()]
        if not cand or not refs:
            continue
        for r in refs:
            flat_cands.append(cand)
            flat_refs.append(r)
            flat_item_idx.append(i)

    if not flat_cands:
        return out

    sums: Dict[int, float] = defaultdict(float)
    cnts: Dict[int, int] = defaultdict(int)

    start = 0
    n = len(flat_cands)
    while start < n:
        end = min(start + max_pairs_per_call, n)
        c_chunk = flat_cands[start:end]
        r_chunk = flat_refs[start:end]
        idx_chunk = flat_item_idx[start:end]

        _, _, F1 = bert_score(
            c_chunk,
            r_chunk,
            lang="en",
            model_type=model_type,
            verbose=False,
            batch_size=batch_size,
        )
        f1_vals = F1.tolist()

        for j, f1 in enumerate(f1_vals):
            ii = idx_chunk[j]
            sums[ii] += float(f1)
            cnts[ii] += 1

        start = end

    for i in range(len(items)):
        if cnts.get(i, 0) > 0:
            out[i] = sums[i] / cnts[i]

    return out


# -------------------------
# Human-proxy evaluation
# -------------------------
# --- Replace evaluate_human_proxy_group() and summarize_by_complexity() ---
# --- And make the small main() adjustments shown at the end. ---

def evaluate_human_proxy_group(
    *,
    model: str,
    dataset: str,
    records: List[Dict[str, Any]],
    qrels_map: Dict[str, Dict[str, float]],
    complexity_map: Optional[Dict[str, str]] = None,
    bert_batch_size: int = 64,
    bert_max_pairs_per_call: int = 3000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    One row per qid (NO persona rows).
    Generation metrics are aggregated across personas for that qid (mean over personas).
    """
    thr = float(REL_THR.get(dataset, 1.0))
    complexity_map = complexity_map or {}

    by_qid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        qid = extract_qid(rec)
        if qid:
            by_qid[qid].append(rec)

    per_q_rows: List[Dict[str, Any]] = []

    # BERTScore: compute per-persona mean BERTScore(answer vs top REF_K docs),
    # then average those per-persona scores into a single per-qid score.
    bert_items: List[Tuple[str, List[str]]] = []
    bert_item_qids: List[str] = []

    # We'll cache per-qid persona metric lists, then fill BERTScore after batching.
    tmp_per_q: Dict[str, Dict[str, Any]] = {}

    for qid, recs in by_qid.items():
        comp = complexity_map.get(qid, "unknown")

        canonical = recs[0]
        sig0 = docid_signature(canonical, k=K)
        for r in recs[1:]:
            if docid_signature(r, k=K) != sig0:
                logger.warning(
                    f"[human_proxy_rag] Retrieval differs across personas for qid={qid}. Using first persona as canonical."
                )
                break

        ranked_docids = extract_ranked_docids(canonical)[:K]
        relmap = qrels_map.get(qid, {})
        if not relmap:
            logger.warning(
                f"[human_proxy_rag] qid={qid} not found in qrels for dataset={dataset}. Retrieval metrics will be 0."
            )

        # retrieval
        acc10 = accuracy_hit_at_10(ranked_docids, relmap, thr)
        rec10 = recall_at_10(ranked_docids, relmap, thr)
        mrr10 = mrr_at_10(ranked_docids, relmap, thr)
        ndcg10 = ndcg_at_10(ranked_docids, relmap)

        evidence_texts = extract_retrieved_texts(canonical, k=K)
        ref_texts = evidence_texts[:REF_K]

        evidence_nuggets: set[str] = set()
        for t in evidence_texts:
            evidence_nuggets.update(nugget_extractor(t))

        # --- aggregate generation across personas ---
        cov_list: List[float] = []
        gfr_list: List[float] = []
        afp_list: List[float] = []
        rouge_list: List[float] = []
        ans_len_list: List[int] = []
        grounded_list: List[int] = []

        for rec in recs:
            answer = extract_answer_text(rec)

            cov, n_grounded, n_answer_sents = grounding_coverage_rouge(
                answer,
                evidence_texts[:REF_K],  # compare vs top REF_K docs
                thr=0.2,
            )
            if cov is not None:
                cov_list.append(float(cov))
            grounded_list.append(int(n_grounded))
            ans_len_list.append(int(n_answer_sents))

                # --- GoldFactRecall_retrieved (NEW RAG STYLE) ---
            gfr, n_gold_nuggets, n_answer_nuggets_set = gold_fact_recall_retrieved(
                answer,
                evidence_texts,
                k=REF_K,
            )

            if gfr is not None:
                gfr_list.append(float(gfr))

            # ROUGE-L vs each ref doc (avg)
            rougeL = None
            if ref_texts and answer:
                rouge_scores = [rouge_l_f1(answer, t) for t in ref_texts]
                rougeL = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
            if rougeL is not None:
                rouge_list.append(float(rougeL))

            # BERTScore item (filled later)
            bert_items.append((answer, ref_texts))
            bert_item_qids.append(qid)

        tmp_per_q[qid] = {
            "model": model,
            "strategy": "human_proxy_rag",
            "dataset": dataset,
            "complexity": comp,
            "qid": qid,
            # retrieval
            "accuracy@10": acc10,
            "recall@10": rec10,
            "mrr@10": mrr10,
            "ndcg@10": ndcg10,
            "n_retrieved": len(ranked_docids),
            "n_relevant": sum(1 for _, rr in relmap.items() if is_rel(float(rr), thr)),
            # evidence stats
            "n_evidence_nuggets": int(len(evidence_nuggets)),
            # generation aggregates (mean over personas; drop missing)
            "coverage": (sum(cov_list) / len(cov_list)) if cov_list else None,
            "gold_fact_recall_retrieved": (sum(gfr_list) / len(gfr_list)) if gfr_list else None,
            "quality_rougeL": (sum(rouge_list) / len(rouge_list)) if rouge_list else None,
            "quality_bertscore": None,  # filled after batching
            # optional diagnostics
            "avg_n_answer_nuggets": (sum(ans_len_list) / len(ans_len_list)) if ans_len_list else None,
            "avg_n_grounded_nuggets": (sum(grounded_list) / len(grounded_list)) if grounded_list else None,
            "n_personas": len(recs),
        }

    # --- batched BERTScore per persona, then mean per qid ---
    if bert_items:
        bert_means = _batched_mean_bertscore_f1(
            bert_items,
            model_type="roberta-base",
            batch_size=int(bert_batch_size),
            max_pairs_per_call=int(bert_max_pairs_per_call),
        )
        per_q_bert_vals: Dict[str, List[float]] = defaultdict(list)
        for qid, b in zip(bert_item_qids, bert_means):
            if b is not None:
                per_q_bert_vals[qid].append(float(b))

        for qid, row in tmp_per_q.items():
            vals = per_q_bert_vals.get(qid, [])
            row["quality_bertscore"] = (sum(vals) / len(vals)) if vals else None

    per_q_rows = list(tmp_per_q.values())
    per_q_df = pd.DataFrame(per_q_rows)

    # overall summary (means over per-qid rows)
    summary: Dict[str, Any] = {
        "model": model,
        "strategy": "human_proxy_rag",
        "dataset": dataset,
        "n_queries": int(per_q_df["qid"].nunique()) if not per_q_df.empty else 0,
    }

    if not per_q_df.empty:
        for col in [
            "accuracy@10",
            "recall@10",
            "mrr@10",
            "ndcg@10",
            "coverage",
            "gold_fact_recall_retrieved",
            "quality_rougeL",
            "quality_bertscore",
        ]:
            vals = per_q_df[col].dropna() if col in per_q_df.columns else []
            summary[col] = float(vals.mean()) if len(vals) else None

        summary["avg_n_evidence_nuggets"] = float(per_q_df["n_evidence_nuggets"].mean())
        if "avg_n_answer_nuggets" in per_q_df.columns:
            summary["avg_n_answer_nuggets"] = float(per_q_df["avg_n_answer_nuggets"].dropna().mean())
        if "n_personas" in per_q_df.columns:
            summary["avg_n_personas"] = float(per_q_df["n_personas"].mean())

    return per_q_df, summary


def summarize_by_complexity(per_q_df: pd.DataFrame) -> pd.DataFrame:
    """
    Complexity-wise summary (one row per qid already).
    Output: one row per (model, dataset, complexity).
    """
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
            coverage=("coverage", "mean"),
            gold_fact_recall_retrieved=("gold_fact_recall_retrieved", "mean"),
            quality_rougeL=("quality_rougeL", "mean"),
            quality_bertscore=("quality_bertscore", "mean"),
            avg_n_evidence_nuggets=("n_evidence_nuggets", "mean"),
            avg_n_answer_nuggets=("avg_n_answer_nuggets", "mean"),
            avg_n_grounded_nuggets=("avg_n_grounded_nuggets", "mean"),
            avg_n_personas=("n_personas", "mean"),
        )
        .reset_index()
    )



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate human_proxy_rag: retrieval metrics once per query + generation metrics per persona (batched BERTScore)"
    )
    parser.add_argument("--inputs", type=str, required=True, help="Directory containing JSONL result files")
    parser.add_argument("--out", type=str, required=True, help="Output CSV file for summary metrics")
    parser.add_argument("--detailed-out", type=str, help="Optional output CSV for detailed rows")
    parser.add_argument("--by-complexity-out", type=str, help="Optional output CSV for complexity-wise metrics")
    parser.add_argument(
        "--datasets",
        type=str,
        default="beir,nq,trecdl2019,trecdl2020,trecdl2021",
        help="Comma-separated dataset keys to include",
    )
    parser.add_argument("--bert-batch-size", type=int, default=64, help="BERTScore internal batch size")
    parser.add_argument(
        "--bert-max-pairs",
        type=int,
        default=3000,
        help="Max (candidate,reference) pairs per bert_score() call (chunking safety)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    allowed = {normalize_dataset_key(d.strip()) for d in args.datasets.split(",") if d.strip()}
    allowed = {d for d in allowed if d is not None}

    input_dir = Path(args.inputs)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    t0 = time.perf_counter()

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for f in sorted(input_dir.glob("*.jsonl")):
        meta = parse_filename_meta(f.name)
        if not meta:
            continue

        ds_norm = normalize_dataset_key(meta["dataset"])
        if ds_norm is None or ds_norm not in allowed:
            continue

        rows = load_jsonl(f)
        rows = [r for r in rows if (r.get("strategy") in (None, "human_proxy_rag"))]  # strict human_proxy_rag-only

        grouped[(meta["model"], ds_norm)].extend(rows)

    if not grouped:
        logger.error("No human_proxy_rag results found in inputs.")
        return

    qrels_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
    complexity_cache: Dict[str, Dict[str, str]] = {}

    summary_rows: List[Dict[str, Any]] = []
    detailed_frames: List[pd.DataFrame] = []

    for (model, dataset), records in grouped.items():
        tg0 = time.perf_counter()
        logger.info(f"Evaluating human_proxy_rag: model={model} dataset={dataset} n_records={len(records)}")

        if dataset not in qrels_cache:
            bundle = load_bundle(dataset)
            qrels_cache[dataset] = make_qrels_map(bundle.qrels_df)
            complexity_cache[dataset] = make_complexity_map(bundle.queries_df)

        detailed_df, summary = evaluate_human_proxy_group(
            model=model,
            dataset=dataset,
            records=records,
            qrels_map=qrels_cache[dataset],
            complexity_map=complexity_cache.get(dataset, {}),
            bert_batch_size=args.bert_batch_size,
            bert_max_pairs_per_call=args.bert_max_pairs,
        )

        summary_rows.append(summary)
        if not detailed_df.empty:
            detailed_frames.append(detailed_df)

        logger.info(f"  done in {time.perf_counter() - tg0:.2f}s; n_queries={summary.get('n_queries', 0)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_path, index=False)
    logger.info(f"Saved summary to {out_path}")

    full_detailed_df = pd.DataFrame()
    if detailed_frames:
        full_detailed_df = pd.concat(detailed_frames, ignore_index=True)

    if args.detailed_out:
        detailed_path = Path(args.detailed_out)
        detailed_path.parent.mkdir(parents=True, exist_ok=True)
        full_detailed_df.to_csv(detailed_path, index=False)
        logger.info(f"Saved detailed to {detailed_path}")

    if args.by_complexity_out:
        byc_df = summarize_by_complexity(full_detailed_df)
        byc_path = Path(args.by_complexity_out)
        byc_path.parent.mkdir(parents=True, exist_ok=True)
        byc_df.to_csv(byc_path, index=False)
        logger.info(f"Saved by-complexity to {byc_path}")

    print("\nHuman-Proxy RAG Evaluation Summary (@10 + generation):")
    print("=" * 95)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No results to display")

    if args.by_complexity_out and not full_detailed_df.empty:
        byc_df = summarize_by_complexity(full_detailed_df)
        if not byc_df.empty:
            print("\nHuman-Proxy RAG Evaluation By Complexity (@10 + generation):")
            print("=" * 95)
            print(byc_df.to_string(index=False))

    logger.info(f"Total eval time: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
