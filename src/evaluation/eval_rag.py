
# #With GOLD FACTS-ORIGINAL
# # src/evaluation/eval_rag.py
# import transformers
# transformers.logging.set_verbosity_error()

# import argparse
# import json
# import logging
# import math
# import re
# import time
# from collections import defaultdict
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import pandas as pd
# from bert_score import score as bert_score

# from src.managers.dataset_manager import DatasetManager

# logger = logging.getLogger(__name__)

# # Evidence window / retrieval @K
# K = 10
# REF_K = 10
# REL_THRESHOLD = 1.0  # relevant if rel >= 1 for binary metrics; nDCG uses graded rel.

# # -------------------------
# # RAG-ONLY filename matcher
# # -------------------------
# FILENAME_RE = re.compile(
#     r"^(?P<model>.+?)__rag__(?P<dataset>.+?)__.*\.jsonl$"
# )

# # -------------------------
# # IO
# # -------------------------
# def load_jsonl(path: Path) -> List[Dict[str, Any]]:
#     rows = []
#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             if line.strip():
#                 rows.append(json.loads(line))
#     return rows


# def parse_filename_meta(name: str) -> Dict[str, str]:
#     m = FILENAME_RE.match(name)
#     if not m:
#         return {}
#     return {"model": m.group("model"), "strategy": "rag", "dataset": m.group("dataset")}

# # -------------------------
# # Dataset key normalization
# # -------------------------
# def normalize_dataset_key(ds: str) -> Optional[str]:
#     """
#     Normalize to one of:
#       {beir, nq, trecdl2019, trecdl2020, trecdl2021}
#     """
#     if not ds:
#         return None
#     s = str(ds).strip().lower().replace("-", "_")

#     for suf in ["_judged", "_full"]:
#         if s.endswith(suf):
#             s = s[: -len(suf)]

#     if s.startswith("trecdl_"):
#         s = s.replace("trecdl_", "trecdl")

#     if s.startswith("trecdl") and any(y in s for y in ["2019", "2020", "2021"]):
#         m = re.search(r"(2019|2020|2021)", s)
#         if m:
#             return f"trecdl{m.group(1)}"

#     if s in {"beir", "scidocs", "beir_scidocs"}:
#         return "beir"
#     if s in {"nq", "natural_questions", "naturalquestion"}:
#         return "nq"

#     return None


# # -------------------------
# # Dataset loading + qrels map
# # -------------------------
# def load_bundle(dataset_key: str):
#     dm = DatasetManager()

#     if dataset_key.startswith("trecdl"):
#         year = int(dataset_key.replace("trecdl", ""))  # trecdl2021 -> 2021
#         bundle, _ = dm.load_bundle(f"trecdl_{year}", cfg={}, limit=None)
#         return bundle

#     if dataset_key == "beir":
#         bundle, _ = dm.load_bundle("beir", cfg={}, limit=None)
#         return bundle

#     if dataset_key == "nq":
#         bundle, _ = dm.load_bundle("nq", cfg={}, limit=None)
#         return bundle

#     raise ValueError(f"Unsupported dataset key: {dataset_key}")


# def make_qrels_map(qrels_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
#     qcol = "qid" if "qid" in qrels_df.columns else ("query_id" if "query_id" in qrels_df.columns else None)
#     dcol = "docid" if "docid" in qrels_df.columns else ("doc_id" if "doc_id" in qrels_df.columns else None)
#     rcol = next((c for c in ["relevance", "rel", "score", "label", "judgment"] if c in qrels_df.columns), None)

#     if qcol is None or dcol is None or rcol is None:
#         raise ValueError(f"qrels_df needs qid/docid/relevance-ish columns. Found: {list(qrels_df.columns)}")

#     mp: Dict[str, Dict[str, float]] = {}
#     for _, row in qrels_df.iterrows():
#         qid = str(row[qcol])
#         docid = str(row[dcol])
#         rel = float(row[rcol])
#         mp.setdefault(qid, {})[docid] = rel
#     return mp


# def make_complexity_map(queries_df: pd.DataFrame) -> Dict[str, str]:
#     """
#     Build qid -> complexity label map (for complexity-wise reporting).

#     Tries these columns:
#       qid: "qid" or "query_id"
#       complexity: "complexity", "complexity_label", "difficulty", "label"
#     """
#     qcol = "qid" if "qid" in queries_df.columns else ("query_id" if "query_id" in queries_df.columns else None)
#     ccol = next((c for c in ["complexity", "complexity_label", "difficulty", "label"] if c in queries_df.columns), None)

#     if qcol is None or ccol is None:
#         logger.warning(
#             f"queries_df missing qid/complexity columns; complexity-wise metrics will be 'unknown'. "
#             f"Found: {list(queries_df.columns)}"
#         )
#         return {}

#     mp: Dict[str, str] = {}
#     for _, row in queries_df.iterrows():
#         qid = str(row[qcol])
#         comp = row.get(ccol)
#         comp_s = "" if comp is None else str(comp).strip()
#         if comp_s:
#             mp[qid] = comp_s
#     return mp


# def is_rel(rel: float, thr: float = REL_THRESHOLD) -> bool:
#     return rel >= thr


# # -------------------------
# # Retrieval metrics @10
# # -------------------------
# def accuracy_hit_at_10(ranked: List[str], relmap: Dict[str, float]) -> float:
#     return 1.0 if any(d in relmap and is_rel(relmap[d]) for d in ranked[:K]) else 0.0


# def recall_at_10(ranked: List[str], relmap: Dict[str, float]) -> float:
#     relevant = [d for d, r in relmap.items() if is_rel(r)]
#     if not relevant:
#         return 0.0
#     hits = sum(1 for d in ranked[:K] if d in relmap and is_rel(relmap[d]))
#     return hits / len(relevant)


# def mrr_at_10(ranked: List[str], relmap: Dict[str, float]) -> float:
#     for i, d in enumerate(ranked[:K], start=1):
#         if d in relmap and is_rel(relmap[d]):
#             return 1.0 / i
#     return 0.0


# def ndcg_at_10(ranked: List[str], relmap: Dict[str, float]) -> float:
#     def dcg(vals: List[float]) -> float:
#         s = 0.0
#         for i, v in enumerate(vals, start=1):
#             s += (2.0 ** v - 1.0) / math.log2(i + 1.0)
#         return s

#     gains = [float(relmap.get(d, 0.0)) for d in ranked[:K]]
#     ideal = sorted([float(r) for r in relmap.values()], reverse=True)[:K]
#     denom = dcg(ideal)
#     return 0.0 if denom == 0.0 else dcg(gains) / denom


# # -------------------------
# # Extraction helpers
# # -------------------------
# def extract_answer_text(rec: Dict[str, Any]) -> str:
#     for k in ["answer", "final_answer", "response", "generation", "generated_answer", "output"]:
#         v = rec.get(k)
#         if isinstance(v, str) and v.strip():
#             return v.strip()
#     return ""


# def extract_qid(rec: Dict[str, Any]) -> Optional[str]:
#     v = rec.get("qid") or rec.get("query_id") or rec.get("q_id") or rec.get("id")
#     return None if v is None else str(v)


# def extract_retrieved_items(rec: Dict[str, Any], k: int = K) -> List[Tuple[int, str, str]]:
#     """
#     Returns list of (rank, docid, text) from rec["retrieved_docs"].
#     """
#     docs = rec.get("retrieved_docs")
#     if not isinstance(docs, list):
#         return []

#     out: List[Tuple[int, str, str]] = []
#     for i, item in enumerate(docs):
#         if not isinstance(item, dict):
#             continue

#         rank = int(item.get("rank", i + 1))

#         docid = (
#             item.get("docid")
#             or item.get("doc_id")
#             or item.get("docno")
#             or item.get("pid")
#             or item.get("id")
#         )
#         docid = "" if docid is None else str(docid)

#         txt = (
#             item.get("passage")
#             or item.get("text")
#             or item.get("contents")
#             or item.get("body")
#             or item.get("document")
#             or ""
#         )
#         txt = "" if txt is None else str(txt)

#         # Keep only useful rows
#         if docid or txt:
#             out.append((rank, docid, txt))

#     out.sort(key=lambda x: x[0])
#     return out[:k]


# # -------------------------
# # Nugget extraction
# # -------------------------
# _SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# _WS_RE = re.compile(r"\s+")
# _PUNCT_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$")


# def normalize_nugget(s: str) -> str:
#     s = s.lower().strip()
#     s = _WS_RE.sub(" ", s)
#     return _PUNCT_EDGE_RE.sub("", s)


# def nugget_extractor(text: str) -> List[str]:
#     if not text:
#         return []
#     parts = _SENT_SPLIT_RE.split(text)
#     nuggets: List[str] = []
#     seen = set()
#     for sent in parts:
#         n = normalize_nugget(sent)
#         if len(n) >= 20 and n not in seen:
#             nuggets.append(n)
#             seen.add(n)
#     return nuggets


# # -------------------------
# # GoldFactRecall_retrieved (NEW)
# # -------------------------
# def gold_fact_recall_retrieved(
#     answer: str,
#     retrieved_texts: List[str],
#     *,
#     k: int = REF_K,
# ) -> Tuple[Optional[float], int, int]:
#     """
#     GoldFactRecall_retrieved = |N_answer ∩ N_gold| / |N_gold|,
#     where N_gold are nuggets extracted from the retrieved documents.

#     Returns:
#       (score or None, n_gold_nuggets, n_answer_nuggets_set)
#     """
#     if not answer:
#         return None, 0, 0

#     gold_nuggets = set().union(*(nugget_extractor(t) for t in (retrieved_texts or [])[:k]))
#     if not gold_nuggets:
#         return None, 0, len(set(nugget_extractor(answer)))

#     answer_nuggets = set(nugget_extractor(answer))
#     score = len(answer_nuggets & gold_nuggets) / len(gold_nuggets)
#     return score, len(gold_nuggets), len(answer_nuggets)


# # -------------------------
# # ROUGE-L
# # -------------------------
# _TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# def lcs_len(a: List[str], b: List[str]) -> int:
#     dp = [0] * (len(b) + 1)
#     for i in range(1, len(a) + 1):
#         prev = 0
#         for j in range(1, len(b) + 1):
#             cur = dp[j]
#             dp[j] = prev + 1 if a[i - 1] == b[j - 1] else max(dp[j], dp[j - 1])
#             prev = cur
#     return dp[-1]


# def rouge_l_f1(candidate: str, reference: str) -> float:
#     c = _TOKEN_RE.findall(candidate.lower())
#     r = _TOKEN_RE.findall(reference.lower())
#     if not c or not r:
#         return 0.0
#     lcs = lcs_len(c, r)
#     p, q = lcs / len(c), lcs / len(r)
#     return 0.0 if p + q == 0 else 2 * p * q / (p + q)

# def grounding_coverage_rouge(
#     answer: str,
#     evidence_texts: List[str],
#     *,
#     thr: float = 0.2,
# ) -> Tuple[Optional[float], int, int]:
#     """
#     Returns:
#       coverage (0..1 or None), grounded_count, total_answer_sents

#     Rule:
#       answer sentence is grounded if max_{doc in evidence} ROUGE-L(sentence, doc) >= thr
#     """
#     ans_sents = nugget_extractor(answer) # returns List[str]
#     if not ans_sents:
#         return None, 0, 0

#     ev = [normalize_nugget(t) for t in (evidence_texts or []) if isinstance(t, str) and t.strip()]
#     if not ev:
#         return None, 0, len(ans_sents)

#     grounded = 0
#     for s in ans_sents:
#         best = 0.0
#         for d in ev:
#             best = max(best, rouge_l_f1(s, d))
#             if best >= thr:
#                 break
#         if best >= thr:
#             grounded += 1

#     return grounded / len(ans_sents), grounded, len(ans_sents)


# # -------------------------
# # BERTScore (BATCHED)
# # -------------------------
# def batched_bertscore_f1(pairs: List[Tuple[str, str]], batch_size: int = 64) -> List[float]:
#     out = []
#     for i in range(0, len(pairs), batch_size):
#         chunk = pairs[i : i + batch_size]
#         cands = [c for c, _ in chunk]
#         refs = [r for _, r in chunk]
#         _, _, F1 = bert_score(
#             cands,
#             refs,
#             lang="en",
#             model_type="roberta-base",
#             verbose=False,
#             batch_size=batch_size,
#         )
#         out.extend([float(x) for x in F1])
#     return out


# # -------------------------
# # Evaluation (RAG: retrieval + generation)
# # -------------------------
# def evaluate_group_rag(
#     *,
#     model: str,
#     dataset: str,
#     records: List[Dict[str, Any]],
#     qrels_map: Dict[str, Dict[str, float]],
#     complexity_map: Optional[Dict[str, str]] = None,
# ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
#     per_q_rows = []
#     bert_pairs: List[Tuple[str, str]] = []
#     bert_spans: Dict[str, Tuple[int, int]] = {}

#     complexity_map = complexity_map or {}

#     for rec in records:
#         qid = extract_qid(rec)
#         if not qid:
#             continue

#         # complexity exists only in the JSONL record
#         comp = rec["complexity"] if "complexity" in rec else "unknown"

#         answer = extract_answer_text(rec)
#         retrieved_items = extract_retrieved_items(rec, k=max(K, REF_K))
#         ranked_docids = [d for _, d, _ in retrieved_items if d][:K]
#         retrieved_texts = [t for _, _, t in retrieved_items if t]

#         # -------------------------
#         # Retrieval metrics @10
#         # -------------------------
#         relmap = qrels_map.get(qid, {})
#         if not relmap:
#             logger.warning(f"[rag] qid={qid} not found in qrels for dataset={dataset}. Retrieval metrics will be 0.")
#         acc10 = accuracy_hit_at_10(ranked_docids, relmap) if ranked_docids else None
#         rec10 = recall_at_10(ranked_docids, relmap) if ranked_docids else None
#         mrr10 = mrr_at_10(ranked_docids, relmap) if ranked_docids else None
#         ndcg10 = ndcg_at_10(ranked_docids, relmap) if ranked_docids else None

#         # -------------------------
#         # Evidence/generation metrics
#         # -------------------------
#         evidence_nuggets = set().union(*(nugget_extractor(t) for t in retrieved_texts[:K]))

#         # Existing grounding coverage (answer -> evidence, ROUGE sentence grounding)
#         cov, n_grounded, n_answer_sents = grounding_coverage_rouge(
#             answer,
#             retrieved_texts[:REF_K],
#             thr=0.2,
#         )

#         # NEW: GoldFactRecall_retrieved (evidence -> answer, nugget recall)
#         gfr, n_gold_nuggets, n_answer_nuggets_set = gold_fact_recall_retrieved(
#             answer,
#             retrieved_texts,
#             k=REF_K,
#         )

#         rougeL = (
#             sum(rouge_l_f1(answer, t) for t in retrieved_texts[:REF_K]) / len(retrieved_texts[:REF_K])
#             if answer and retrieved_texts
#             else None
#         )

#         if answer and retrieved_texts:
#             start = len(bert_pairs)
#             for t in retrieved_texts[:REF_K]:
#                 bert_pairs.append((answer, t))
#             bert_spans[qid] = (start, len(bert_pairs))

#         per_q_rows.append(
#             {
#                 "model": model,
#                 "strategy": "rag",
#                 "dataset": dataset,
#                 "complexity": comp,
#                 "qid": qid,
#                 # retrieval
#                 "accuracy@10": acc10,
#                 "recall@10": rec10,
#                 "mrr@10": mrr10,
#                 "ndcg@10": ndcg10,
#                 "n_candidates": len(ranked_docids),
#                 "n_relevant": sum(1 for _, r in relmap.items() if is_rel(r)),
#                 "top10_docids": " ".join(ranked_docids[:K]),
#                 # generation/evidence
#                 "coverage": cov,
#                 "n_grounded_nuggets": n_grounded,
#                 "n_answer_nuggets": n_answer_sents,
#                 "gold_fact_recall_retrieved": gfr,
#                 "n_gold_nuggets": n_gold_nuggets,
#                 "n_answer_nuggets_set": n_answer_nuggets_set,
#                 "quality_rougeL": rougeL,
#                 "quality_bertscore": None,
#                 "n_retrieved_docs": len(retrieved_texts),
#                 "n_evidence_nuggets": len(evidence_nuggets),
#             }
#         )

#     # --- batched BERTScore ---
#     if bert_pairs:
#         f1s = batched_bertscore_f1(bert_pairs, batch_size=64)
#         for row in per_q_rows:
#             qid = row["qid"]
#             if qid in bert_spans:
#                 s, e = bert_spans[qid]
#                 row["quality_bertscore"] = sum(f1s[s:e]) / (e - s)

#     df = pd.DataFrame(per_q_rows)

#     summary = dict(
#         model=model,
#         strategy="rag",
#         dataset=dataset,
#         n_queries=int(df["qid"].nunique()) if not df.empty else 0,
#     )

#     if not df.empty:
#         # retrieval means
#         summary["accuracy@10"] = float(df["accuracy@10"].mean())
#         summary["recall@10"] = float(df["recall@10"].mean())
#         summary["mrr@10"] = float(df["mrr@10"].mean())
#         summary["ndcg@10"] = float(df["ndcg@10"].mean())
#         # generation means
#         summary["coverage"] = float(df["coverage"].mean())
#         summary["gold_fact_recall_retrieved"] = float(df["gold_fact_recall_retrieved"].mean())
#         summary["quality_rougeL"] = float(df["quality_rougeL"].mean())
#         summary["quality_bertscore"] = float(df["quality_bertscore"].mean())
#         summary["avg_n_evidence_nuggets"] = float(df["n_evidence_nuggets"].mean())
#         summary["avg_n_gold_nuggets"] = float(df["n_gold_nuggets"].mean())
#         summary["avg_n_answer_nuggets"] = float(df["n_answer_nuggets"].mean())
#         summary["avg_n_answer_nuggets_set"] = float(df["n_answer_nuggets_set"].mean())
#         summary["avg_n_retrieved_docs"] = float(df["n_retrieved_docs"].mean())

#     return df, summary

# def summarize_by_complexity(per_q_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Complexity-wise mean metrics, grouped by (model, dataset, complexity).
#     """
#     if per_q_df.empty or "complexity" not in per_q_df.columns:
#         return pd.DataFrame()

#     agg = (
#         per_q_df.groupby(["model", "dataset", "complexity"], dropna=False)
#         .agg(
#             n_queries=("qid", "nunique"),
#             accuracy_at_10=("accuracy@10", "mean"),
#             recall_at_10=("recall@10", "mean"),
#             mrr_at_10=("mrr@10", "mean"),
#             ndcg_at_10=("ndcg@10", "mean"),
#             coverage=("coverage", "mean"),
#             gold_fact_recall_retrieved=("gold_fact_recall_retrieved", "mean"),
#             quality_rougeL=("quality_rougeL", "mean"),
#             quality_bertscore=("quality_bertscore", "mean"),
#             avg_n_evidence_nuggets=("n_evidence_nuggets", "mean"),
#             avg_n_gold_nuggets=("n_gold_nuggets", "mean"),
#             avg_n_answer_nuggets=("n_answer_nuggets", "mean"),
#             avg_n_answer_nuggets_set=("n_answer_nuggets_set", "mean"),
#             avg_n_retrieved_docs=("n_retrieved_docs", "mean"),
#         )
#         .reset_index()
#     )
#     return agg


# def main():
#     parser = argparse.ArgumentParser(description="Evaluate RAG: retrieval metrics + evidence/generation metrics")
#     parser.add_argument("--inputs", required=True, help="Directory containing JSONL result files")
#     parser.add_argument("--out", required=True, help="Output CSV file for summary metrics")
#     parser.add_argument("--detailed-out", help="Optional output CSV for per-query metrics")
#     parser.add_argument("--by-complexity-out", help="Optional output CSV for complexity-wise metrics")
#     parser.add_argument("--datasets", default="beir,nq,trecdl2019,trecdl2020,trecdl2021")
#     args = parser.parse_args()

#     logging.basicConfig(level=logging.INFO)

#     allowed = {normalize_dataset_key(d.strip()) for d in args.datasets.split(",") if d.strip()}
#     allowed = {d for d in allowed if d is not None}

#     input_dir = Path(args.inputs)
#     if not input_dir.exists():
#         logger.error(f"Input directory does not exist: {input_dir}")
#         return

#     t0 = time.perf_counter()

#     # 1) Group records
#     grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

#     for f in input_dir.glob("*.jsonl"):
#         meta = parse_filename_meta(f.name)
#         if not meta:
#             continue

#         ds_norm = normalize_dataset_key(meta["dataset"])
#         if ds_norm is None or ds_norm not in allowed:
#             continue

#         rows = load_jsonl(f)
#         rows = [r for r in rows if (r.get("strategy") in (None, "rag"))]  # strict rag-only
#         grouped[(meta["model"], ds_norm)].extend(rows)

#     if not grouped:
#         logger.error("No RAG results found in inputs.")
#         return

#     # 2) Cache qrels + complexity per dataset
#     qrels_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
#     complexity_cache: Dict[str, Dict[str, str]] = {}

#     summaries: List[Dict[str, Any]] = []
#     detailed_frames: List[pd.DataFrame] = []

#     for (model, dataset), records in grouped.items():
#         tg0 = time.perf_counter()
#         logger.info(f"Evaluating RAG: model={model} dataset={dataset} n_rows={len(records)}")

#         if dataset not in qrels_cache:
#             bundle = load_bundle(dataset)
#             qrels_cache[dataset] = make_qrels_map(bundle.qrels_df)
#             complexity_cache[dataset] = make_complexity_map(bundle.queries_df)

#         per_q_df, summ = evaluate_group_rag(
#             model=model,
#             dataset=dataset,
#             records=records,
#             qrels_map=qrels_cache[dataset],
#             complexity_map=complexity_cache.get(dataset, {}),
#         )

#         summaries.append(summ)
#         if not per_q_df.empty:
#             detailed_frames.append(per_q_df)

#         logger.info(f"  done in {time.perf_counter() - tg0:.2f}s; n_queries={summ.get('n_queries', 0)}")

#     # 3) Save outputs
#     out_path = Path(args.out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     summary_df = pd.DataFrame(summaries)
#     summary_df.to_csv(out_path, index=False)
#     logger.info(f"Saved summary to {out_path}")

#     detailed_df = pd.DataFrame()
#     if detailed_frames:
#         detailed_df = pd.concat(detailed_frames, ignore_index=True)

#     if args.detailed_out:
#         detailed_path = Path(args.detailed_out)
#         detailed_path.parent.mkdir(parents=True, exist_ok=True)
#         detailed_df.to_csv(detailed_path, index=False)
#         logger.info(f"Saved detailed to {detailed_path}")

#     if args.by_complexity_out:
#         byc = summarize_by_complexity(detailed_df)
#         byc_path = Path(args.by_complexity_out)
#         byc_path.parent.mkdir(parents=True, exist_ok=True)
#         byc.to_csv(byc_path, index=False)
#         logger.info(f"Saved by-complexity to {byc_path}")

#     # 4) Print summary (like old code)
#     print("\nRAG Evaluation Summary (@10 + generation):")
#     print("=" * 90)
#     if not summary_df.empty:
#         print(summary_df.to_string(index=False))
#     else:
#         print("No results to display")

#     if args.by_complexity_out and not detailed_df.empty:
#         byc = summarize_by_complexity(detailed_df)
#         if not byc.empty:
#             print("\nRAG Evaluation By Complexity (@10 + generation):")
#             print("=" * 90)
#             print(byc.to_string(index=False))

#     logger.info(f"Total eval time: {time.perf_counter() - t0:.2f}s")


# if __name__ == "__main__":
#     main()

##NEW
# src/evaluation/eval_rag.py
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
K = 10                 # retrieval metrics @K
REF_K = 10             # evidence window for generation metrics
REL_THRESHOLD = 1.0    # binary relevant if rel >= 1; nDCG uses graded rel

# -------------------------
# RAG filename matcher
# -------------------------
FILENAME_RE = re.compile(r"^(?P<model>.+?)__rag__(?P<dataset>.+?)__.*\.jsonl$")

# -------------------------
# IO
# -------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_filename_meta(name: str) -> Dict[str, str]:
    m = FILENAME_RE.match(name)
    if not m:
        return {}
    return {"model": m.group("model"), "strategy": "rag", "dataset": m.group("dataset")}


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
    s = str(ds).strip().lower().replace("-", "_")

    for suf in ["_judged", "_full"]:
        if s.endswith(suf):
            s = s[: -len(suf)]

    if s.startswith("trecdl_"):
        s = s.replace("trecdl_", "trecdl")

    if s.startswith("trecdl") and any(y in s for y in ["2019", "2020", "2021"]):
        m = re.search(r"(2019|2020|2021)", s)
        if m:
            return f"trecdl{m.group(1)}"

    if s in {"beir", "scidocs", "beir_scidocs"}:
        return "beir"
    if s in {"nq", "natural_questions", "naturalquestion"}:
        return "nq"

    return None


# -------------------------
# Dataset loading + qrels map
# -------------------------
def load_bundle(dataset_key: str):
    dm = DatasetManager()

    if dataset_key.startswith("trecdl"):
        year = int(dataset_key.replace("trecdl", ""))
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
        rel = float(row[rcol])   # <-- FIX (was row["relevance"])
        mp.setdefault(qid, {})[docid] = rel
    return mp


def make_complexity_map(queries_df: pd.DataFrame) -> Dict[str, str]:
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
# Extraction helpers
# -------------------------
def extract_answer_text(rec: Dict[str, Any]) -> str:
    for k in ["answer", "final_answer", "response", "generation", "generated_answer", "output"]:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def extract_qid(rec: Dict[str, Any]) -> Optional[str]:
    v = rec.get("qid") or rec.get("query_id") or rec.get("q_id") or rec.get("id")
    return None if v is None else str(v)


def extract_retrieved_items(rec: Dict[str, Any]) -> List[Tuple[int, str, str]]:
    """
    Returns list of (rank, docid, text) from rec["retrieved_docs"] in their given order/rank.
    """
    docs = rec.get("retrieved_docs")
    if not isinstance(docs, list):
        return []

    out: List[Tuple[int, str, str]] = []
    for i, item in enumerate(docs):
        if not isinstance(item, dict):
            continue

        rank = item.get("rank")
        rank = int(rank) if isinstance(rank, (int, float, str)) and str(rank).strip().isdigit() else (i + 1)

        docid = (
            item.get("docid")
            or item.get("doc_id")
            or item.get("docno")
            or item.get("pid")
            or item.get("id")
        )
        docid = "" if docid is None else str(docid)

        txt = (
            item.get("passage")
            or item.get("text")
            or item.get("contents")
            or item.get("body")
            or item.get("document")
            or ""
        )
        txt = "" if txt is None else str(txt)

        if docid or txt:
            out.append((rank, docid, txt))

    out.sort(key=lambda x: x[0])
    return out


def merge_dedup_ranking(items: List[Tuple[int, str, str]]) -> Tuple[List[str], List[str]]:
    """
    Standard IR ranking:
      - sort by rank (already sorted upstream, but kept safe)
      - deduplicate docids (keep first occurrence)
    Returns:
      ranked_docids, ranked_texts (aligned to the kept docids)
    """
    items = sorted(items, key=lambda x: x[0])
    seen = set()
    docids: List[str] = []
    texts: List[str] = []
    for _, docid, txt in items:
        if not docid:
            continue
        if docid in seen:
            continue
        seen.add(docid)
        docids.append(docid)
        texts.append(txt or "")
    return docids, texts


# -------------------------
# Nugget extraction (≥3 words)
# -------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WS_RE = re.compile(r"\s+")
_PUNCT_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$")


def normalize_nugget(s: str) -> str:
    s = s.lower().strip()
    s = _WS_RE.sub(" ", s)
    s = _PUNCT_EDGE_RE.sub("", s)
    return s.strip()


def nugget_extractor(text: str, *, min_words: int = 3) -> List[str]:
    """
    Shared nugget rule across RAG + hierarchy:
      keep nuggets with >= min_words tokens (default 3).
    """
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


# -------------------------
# GoldFactRecall_retrieved (NO ROUGE)
# -------------------------
def gold_fact_recall_retrieved(
    answer: str,
    retrieved_texts: List[str],
    *,
    k: int = REF_K,
) -> Tuple[Optional[float], int, int]:
    """
    GoldFactRecall_retrieved = |N_answer ∩ N_gold| / |N_gold|,
    where N_gold are nuggets extracted from the retrieved evidence texts.

    NOTE: No ROUGE is used here (pure nugget set overlap).
    """
    if not answer:
        return None, 0, 0

    gold_nuggets = set().union(*(nugget_extractor(t) for t in (retrieved_texts or [])[:k]))
    answer_nuggets = set(nugget_extractor(answer))

    if not gold_nuggets:
        return None, 0, len(answer_nuggets)

    score = len(answer_nuggets & gold_nuggets) / len(gold_nuggets)
    return score, len(gold_nuggets), len(answer_nuggets)


# -------------------------
# ROUGE-L (used for grounding coverage + optional quality)
# -------------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


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
    thr: float = 0.2,
) -> Tuple[Optional[float], int, int]:
    """
    Coverage = grounded_answer_nuggets / answer_nuggets

    Grounding rule:
      an answer nugget is grounded if max_{evidence_nugget} ROUGE-L(answer_nugget, evidence_nugget) >= thr

    IMPORTANT CHANGE:
      evidence is nuggetized (not whole documents).
    """
    ans_nuggets = nugget_extractor(answer)
    if not ans_nuggets:
        return None, 0, 0

    ev_nuggets = set().union(*(nugget_extractor(t) for t in (evidence_texts or []) if isinstance(t, str) and t.strip()))
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
# BERTScore (batched)
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
# Evaluation (RAG)
# -------------------------
def evaluate_group_rag(
    *,
    model: str,
    dataset: str,
    records: List[Dict[str, Any]],
    qrels_map: Dict[str, Dict[str, float]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    per_q_rows = []
    bert_pairs: List[Tuple[str, str]] = []
    bert_spans: Dict[str, Tuple[int, int]] = {}

    for rec in records:
        qid = extract_qid(rec)
        if not qid:
            continue

        comp = rec["complexity"] if "complexity" in rec else "unknown"
        answer = extract_answer_text(rec)

        retrieved_items = extract_retrieved_items(rec)

        # CHANGE (3) + (4): merged+dedup ranking built ONLY from what RAG outputs
        ranked_docids_all, ranked_texts_all = merge_dedup_ranking(retrieved_items)

        # Retrieval metrics @K on merged+dedup ranking
        relmap = qrels_map.get(qid, {})
        if not relmap:
            logger.warning(f"[rag] qid={qid} not found in qrels for dataset={dataset}. Retrieval metrics will be 0/None.")

        acc_k = accuracy_hit_at_k(ranked_docids_all, relmap, k=K) if ranked_docids_all else None
        rec_k = recall_at_k(ranked_docids_all, relmap, k=K) if ranked_docids_all else None
        mrr_k = mrr_at_k(ranked_docids_all, relmap, k=K) if ranked_docids_all else None
        ndcg_k = ndcg_at_k(ranked_docids_all, relmap, k=K) if ranked_docids_all else None

        # Evidence window: again, ONLY from what RAG outputs
        evidence_texts = ranked_texts_all[:REF_K]
        evidence_nuggets = set().union(*(nugget_extractor(t) for t in ranked_texts_all[:K]))

        # CHANGE (1): coverage uses evidence nuggets
        cov, n_grounded, n_answer_nuggets = grounding_coverage_rouge(
            answer,
            evidence_texts,
            thr=0.2,
        )

        # CHANGE (5): still no ROUGE here
        gfr, n_gold_nuggets, n_answer_nuggets_set = gold_fact_recall_retrieved(
            answer,
            ranked_texts_all,
            k=REF_K,
        )

        # Optional: ROUGE-L quality vs evidence docs (kept as before; not part of GoldFactRecall)
        rougeL = (
            sum(rouge_l_f1(answer, t) for t in evidence_texts) / len(evidence_texts)
            if answer and evidence_texts
            else None
        )

        if answer and evidence_texts:
            start = len(bert_pairs)
            for t in evidence_texts:
                bert_pairs.append((answer, t))
            bert_spans[qid] = (start, len(bert_pairs))

        per_q_rows.append(
            {
                "model": model,
                "strategy": "rag",
                "dataset": dataset,
                "complexity": comp,
                "qid": qid,
                # retrieval
                "accuracy@10": acc_k,
                "recall@10": rec_k,
                "mrr@10": mrr_k,
                "ndcg@10": ndcg_k,
                "n_candidates": len(ranked_docids_all),
                "n_relevant": sum(1 for _, r in relmap.items() if is_rel(r)),
                "topk_docids": " ".join(ranked_docids_all[:K]),
                # evidence/generation
                "coverage": cov,
                "n_grounded_nuggets": n_grounded,
                "n_answer_nuggets": n_answer_nuggets,
                "gold_fact_recall_retrieved": gfr,
                "n_gold_nuggets": n_gold_nuggets,
                "n_answer_nuggets_set": n_answer_nuggets_set,
                "quality_rougeL": rougeL,
                "quality_bertscore": None,
                "n_retrieved_docs": len(ranked_texts_all),
                "n_evidence_nuggets": len(evidence_nuggets),
            }
        )

    # Batched BERTScore
    if bert_pairs:
        f1s = batched_bertscore_f1(bert_pairs, batch_size=64)
        for row in per_q_rows:
            qid = row["qid"]
            if qid in bert_spans:
                s, e = bert_spans[qid]
                row["quality_bertscore"] = sum(f1s[s:e]) / (e - s)

    df = pd.DataFrame(per_q_rows)

    summary = dict(
        model=model,
        strategy="rag",
        dataset=dataset,
        n_queries=int(df["qid"].nunique()) if not df.empty else 0,
    )

    if not df.empty:
        summary["accuracy@10"] = float(df["accuracy@10"].mean())
        summary["recall@10"] = float(df["recall@10"].mean())
        summary["mrr@10"] = float(df["mrr@10"].mean())
        summary["ndcg@10"] = float(df["ndcg@10"].mean())
        summary["coverage"] = float(df["coverage"].mean())
        summary["gold_fact_recall_retrieved"] = float(df["gold_fact_recall_retrieved"].mean())
        summary["quality_rougeL"] = float(df["quality_rougeL"].mean())
        summary["quality_bertscore"] = float(df["quality_bertscore"].mean())
        summary["avg_n_evidence_nuggets"] = float(df["n_evidence_nuggets"].mean())
        summary["avg_n_gold_nuggets"] = float(df["n_gold_nuggets"].mean())
        summary["avg_n_answer_nuggets"] = float(df["n_answer_nuggets"].mean())
        summary["avg_n_answer_nuggets_set"] = float(df["n_answer_nuggets_set"].mean())
        summary["avg_n_retrieved_docs"] = float(df["n_retrieved_docs"].mean())

    return df, summary


def summarize_by_complexity(per_q_df: pd.DataFrame) -> pd.DataFrame:
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
            coverage=("coverage", "mean"),
            gold_fact_recall_retrieved=("gold_fact_recall_retrieved", "mean"),
            quality_rougeL=("quality_rougeL", "mean"),
            quality_bertscore=("quality_bertscore", "mean"),
            avg_n_evidence_nuggets=("n_evidence_nuggets", "mean"),
            avg_n_gold_nuggets=("n_gold_nuggets", "mean"),
            avg_n_answer_nuggets=("n_answer_nuggets", "mean"),
            avg_n_answer_nuggets_set=("n_answer_nuggets_set", "mean"),
            avg_n_retrieved_docs=("n_retrieved_docs", "mean"),
        )
        .reset_index()
    )
    return agg


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG: retrieval metrics + evidence/generation metrics")
    parser.add_argument("--inputs", required=True, help="Directory containing JSONL result files")
    parser.add_argument("--out", required=True, help="Output CSV file for summary metrics")
    parser.add_argument("--detailed-out", help="Optional output CSV for per-query metrics")
    parser.add_argument("--by-complexity-out", help="Optional output CSV for complexity-wise metrics")
    parser.add_argument("--datasets", default="beir,nq,trecdl2019,trecdl2020,trecdl2021")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    allowed = {normalize_dataset_key(d.strip()) for d in args.datasets.split(",") if d.strip()}
    allowed = {d for d in allowed if d is not None}

    input_dir = Path(args.inputs)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    t0 = time.perf_counter()

    # Group records by (model, dataset)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for f in input_dir.glob("*.jsonl"):
        meta = parse_filename_meta(f.name)
        if not meta:
            continue

        ds_norm = normalize_dataset_key(meta["dataset"])
        if ds_norm is None or ds_norm not in allowed:
            continue

        rows = load_jsonl(f)
        rows = [r for r in rows if (r.get("strategy") in (None, "rag"))]  # rag-only
        grouped[(meta["model"], ds_norm)].extend(rows)

    if not grouped:
        logger.error("No RAG results found in inputs.")
        return

    # Cache qrels per dataset
    qrels_cache: Dict[str, Dict[str, Dict[str, float]]] = {}

    summaries: List[Dict[str, Any]] = []
    detailed_frames: List[pd.DataFrame] = []

    for (model, dataset), records in grouped.items():
        tg0 = time.perf_counter()
        logger.info(f"Evaluating RAG: model={model} dataset={dataset} n_rows={len(records)}")

        if dataset not in qrels_cache:
            bundle = load_bundle(dataset)
            qrels_cache[dataset] = make_qrels_map(bundle.qrels_df)

        per_q_df, summ = evaluate_group_rag(
            model=model,
            dataset=dataset,
            records=records,
            qrels_map=qrels_cache[dataset],
        )

        summaries.append(summ)
        if not per_q_df.empty:
            detailed_frames.append(per_q_df)

        logger.info(f"  done in {time.perf_counter() - tg0:.2f}s; n_queries={summ.get('n_queries', 0)}")

    # Save outputs
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summaries)
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
        byc = summarize_by_complexity(detailed_df)
        byc_path = Path(args.by_complexity_out)
        byc_path.parent.mkdir(parents=True, exist_ok=True)
        byc.to_csv(byc_path, index=False)
        logger.info(f"Saved by-complexity to {byc_path}")

    # Print summary
    print("\nRAG Evaluation Summary (@10 + generation):")
    print("=" * 90)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No results to display")

    if args.by_complexity_out and not detailed_df.empty:
        byc = summarize_by_complexity(detailed_df)
        if not byc.empty:
            print("\nRAG Evaluation By Complexity (@10 + generation):")
            print("=" * 90)
            print(byc.to_string(index=False))

    logger.info(f"Total eval time: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
