# # src/evaluation/eval_hierarchy.py
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

# # -------------------------
# # Config
# # -------------------------
# K = 10                  # retrieval @K
# TOPK_EVID_PER_SUBQ = 10 # evidence pool: take top K per subquery (kept; not used for coverage anymore)
# REL_THRESHOLD = 1.0     # relevant if rel >= 1 for binary metrics; nDCG uses graded rel.

# # -------------------------
# # IO
# # -------------------------
# def load_jsonl(path: Path) -> List[Dict[str, Any]]:
#     rows = []
#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 rows.append(json.loads(line))
#     return rows


# # -------------------------
# # Dataset key normalization (GENERAL)
# # -------------------------
# def normalize_dataset_key(ds: str) -> Optional[str]:
#     """
#     Normalize dataset key into one of:
#       - trecdl2019 / trecdl2020 / trecdl2021
#       - beir_scidocs
#       - beir_nq

#     Accepts metadata.dataset (preferred) or filename fragments.
#     """
#     if not ds:
#         return None

#     s = str(ds).strip().lower().replace("-", "_")

#     for suf in ["_judged", "_full"]:
#         if s.endswith(suf):
#             s = s[: -len(suf)]

#     # ---- TREC DL ----
#     if "trecdl" in s or "trec_dl" in s or "msmarco" in s:
#         if "2019" in s:
#             return "trecdl2019"
#         if "2020" in s:
#             return "trecdl2020"
#         if "2021" in s:
#             return "trecdl2021"

#     # ---- BEIR / SCIDOCS ----
#     if "scidocs" in s:
#         return "beir_scidocs"

#     # ---- BEIR / NQ ----
#     if "beir_nq" in s or ("natural" in s and "question" in s) or s.endswith("_nq") or s == "nq":
#         return "beir_nq"

#     return None

# def load_bundle_from_dataset_key(dataset_key: str):
#     """
#     Loads the right IRDatasetBundle (queries + corpus + qrels) given normalized key.
#     NOTE: DatasetManager.load_bundle() in this repo does NOT take `year=`.
#     """
#     dm = DatasetManager()

#     if dataset_key.startswith("trecdl"):
#         year = dataset_key.replace("trecdl", "")   # "2019"/"2020"/"2021"
#         ds_name = f"trecdl_{year}"                 # matches your configs: trecdl_2019, trecdl_2020, trecdl_2021
#         bundle, _ = dm.load_bundle(ds_name, limit=None)
#         return bundle

#     if dataset_key == "beir_scidocs":
#         # your config key is likely just "beir"
#         bundle, _ = dm.load_bundle("beir", limit=None)
#         return bundle

#     if dataset_key == "beir_nq":
#         # your config key is "nq"
#         bundle, _ = dm.load_bundle("nq", limit=None)
#         return bundle

#     raise ValueError(f"Unsupported dataset_key: {dataset_key}")


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
#     Build qid -> complexity label map.
#     Expects complexity to exist in bundle.queries_df.
#     """
#     qcol = "qid" if "qid" in queries_df.columns else ("query_id" if "query_id" in queries_df.columns else None)
#     if qcol is None:
#         logger.warning(f"queries_df missing qid column. Found: {list(queries_df.columns)}")
#         return {}

#     if "complexity" not in queries_df.columns:
#         logger.warning(f"queries_df missing 'complexity' column. Found: {list(queries_df.columns)}")
#         return {}

#     mp: Dict[str, str] = {}
#     for _, row in queries_df.iterrows():
#         qid = str(row[qcol])
#         comp = row.get("complexity")
#         if comp is not None and str(comp).strip():
#             mp[qid] = str(comp).strip().lower()
#     return mp


# def is_rel(rel: float, thr: float = REL_THRESHOLD) -> bool:
#     return rel >= thr


# # -------------------------
# # Retrieval metrics @K
# # -------------------------
# def accuracy_hit_at_k(ranked: List[str], relmap: Dict[str, float], k: int = K) -> float:
#     return 1.0 if any(d in relmap and is_rel(relmap[d]) for d in ranked[:k]) else 0.0


# def recall_at_k(ranked: List[str], relmap: Dict[str, float], k: int = K) -> float:
#     relevant = [d for d, r in relmap.items() if is_rel(r)]
#     if not relevant:
#         return 0.0
#     hits = sum(1 for d in ranked[:k] if d in relmap and is_rel(relmap[d]))
#     return hits / len(relevant)


# def mrr_at_k(ranked: List[str], relmap: Dict[str, float], k: int = K) -> float:
#     for i, d in enumerate(ranked[:k], start=1):
#         if d in relmap and is_rel(relmap[d]):
#             return 1.0 / i
#     return 0.0


# def ndcg_at_k(ranked: List[str], relmap: Dict[str, float], k: int = K) -> float:
#     def dcg(vals: List[float]) -> float:
#         s = 0.0
#         for i, v in enumerate(vals, start=1):
#             s += (2.0 ** v - 1.0) / math.log2(i + 1.0)
#         return s

#     gains = [float(relmap.get(d, 0.0)) for d in ranked[:k]]
#     ideal = sorted([float(r) for r in relmap.values()], reverse=True)[:k]
#     denom = dcg(ideal)
#     return 0.0 if denom == 0.0 else dcg(gains) / denom


# # -------------------------
# # Extraction helpers (hierarchy agent)
# # -------------------------
# def extract_qid(rec: Dict[str, Any]) -> Optional[str]:
#     v = rec.get("qid") or rec.get("query_id") or rec.get("q_id") or rec.get("id")
#     return None if v is None else str(v)


# def extract_dataset_key(rec: Dict[str, Any]) -> Optional[str]:
#     md = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
#     for v in [md.get("dataset"), rec.get("dataset")]:
#         if isinstance(v, str) and v.strip():
#             return normalize_dataset_key(v)
#     return None


# def extract_answer_text(rec: Dict[str, Any]) -> str:
#     for k in ["final_answer", "answer", "response", "generation", "generated_answer", "output"]:
#         v = rec.get(k)
#         if isinstance(v, str) and v.strip():
#             return v.strip()
#     return ""


# def extract_verified_facts(rec: Dict[str, Any]) -> str:
#     v = rec.get("verified_facts")
#     return v.strip() if isinstance(v, str) and v.strip() else ""


# def extract_retrieved_docs_by_subquery(rec: Dict[str, Any]) -> Dict[str, List[Tuple[int, str, str]]]:
#     """
#     Returns:
#       { subquery_str : [(rank, docid, passage), ...] }
#     based on rec["retrieved_docs"] where each item has source_subquery.
#     """
#     docs = rec.get("retrieved_docs")
#     if not isinstance(docs, list):
#         return {}

#     out: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)

#     for i, item in enumerate(docs):
#         if not isinstance(item, dict):
#             continue

#         subq = item.get("source_subquery") or item.get("subquery") or "UNKNOWN_SUBQUERY"
#         subq = str(subq)

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

#         if docid or txt:
#             out[subq].append((rank, docid, txt))

#     for subq in list(out.keys()):
#         out[subq].sort(key=lambda x: x[0])
#     return dict(out)


# # -------------------------
# # Nugget extraction (cheap)
# # -------------------------
# _SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# _WS_RE = re.compile(r"\s+")
# _PUNCT_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$")


# def normalize_nugget(s: str) -> str:
#     s = s.lower().strip()
#     s = _WS_RE.sub(" ", s)
#     return _PUNCT_EDGE_RE.sub("", s)


# def nugget_extractor(text: str) -> List[str]:
#     """
#     Keep aligned with eval_rag.py:
#     returns a LIST of normalized sentence nuggets (deduped, len>=20).
#     """
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
#     Same definition as eval_rag.py.

#     Returns:
#       coverage (0..1 or None), grounded_count, total_answer_sents

#     Rule:
#       answer sentence is grounded if max_{doc in evidence} ROUGE-L(sentence, doc) >= thr
#     """
#     ans_sents = nugget_extractor(answer)  # List[str]
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
# # BERTScore (minimal)
# # -------------------------
# def batched_bertscore_f1(pairs: List[Tuple[str, str]], batch_size: int = 64) -> List[float]:
#     out: List[float] = []
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
# # Per-qid retrieval aggregation across subqueries
# # -------------------------
# def aggregate_subquery_metrics_max(sub_mets: List[Dict[str, float]]) -> Dict[str, float]:
#     """
#     Aggregate retrieval metrics across the 3 subquery ranked lists by taking max.
#     """
#     if not sub_mets:
#         return {"accuracy@10": 0.0, "recall@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0}

#     return {
#         "accuracy@10": max(m["accuracy@10"] for m in sub_mets),
#         "recall@10": max(m["recall@10"] for m in sub_mets),
#         "mrr@10": max(m["mrr@10"] for m in sub_mets),
#         "ndcg@10": max(m["ndcg@10"] for m in sub_mets),
#     }


# # -------------------------
# # Evaluation (Hierarchy agents: any dataset with qrels)
# # -------------------------
# def evaluate_hierarchy(
#     *,
#     model: str,
#     dataset: str,
#     records: List[Dict[str, Any]],
#     qrels_map: Dict[str, Dict[str, float]],
#     complexity_map: Optional[Dict[str, str]] = None,
# ) -> Tuple[pd.DataFrame, Dict[str, Any]]:

#     per_q_rows: List[Dict[str, Any]] = []

#     complexity_map = complexity_map or {}

#     bert_pairs: List[Tuple[str, str]] = []
#     bert_qids: List[str] = []

#     for rec in records:
#         qid = extract_qid(rec)
#         if not qid:
#             continue

#         comp = complexity_map.get(qid, "unknown")
#         answer = extract_answer_text(rec)
#         verified = extract_verified_facts(rec)

#         by_subq = extract_retrieved_docs_by_subquery(rec)
#         relmap = qrels_map.get(qid, {})

#         # --- retrieval metrics per subquery (then max aggregate) ---
#         subquery_metrics: List[Dict[str, float]] = []
#         for subq, items in by_subq.items():
#             ranked_docids = [docid for _, docid, _ in items if docid][:K]
#             if not ranked_docids:
#                 continue

#             if not relmap:
#                 acc10 = rec10 = mrr10 = ndcg10 = 0.0
#             else:
#                 acc10 = accuracy_hit_at_k(ranked_docids, relmap, k=K)
#                 rec10 = recall_at_k(ranked_docids, relmap, k=K)
#                 mrr10 = mrr_at_k(ranked_docids, relmap, k=K)
#                 ndcg10 = ndcg_at_k(ranked_docids, relmap, k=K)

#             subquery_metrics.append(
#                 {
#                     "subquery": subq,
#                     "accuracy@10": acc10,
#                     "recall@10": rec10,
#                     "mrr@10": mrr10,
#                     "ndcg@10": ndcg10,
#                     "top10_docids": " ".join(ranked_docids),
#                 }
#             )

#         agg = aggregate_subquery_metrics_max(subquery_metrics)

#         # -------------------------
#         # Evidence/generation metrics (ALIGN with eval_rag.py)
#         # Coverage is computed ONLY from verified_facts vs final answer.
#         # -------------------------
#         verified_sents = nugget_extractor(verified) if verified else []
#         n_gold = len(verified_sents)

#         cov, n_grounded, n_answer_sents = grounding_coverage_rouge(
#             answer,
#             verified_sents,
#             thr=0.2,
#         )

#         # Align naming with eval_human_proxy_rag.py
#         # Metric B (answer denom): grounded / |answer|
#         answer_fact_precision_retrieved = cov  # = n_grounded / n_answer_sents (or None)

#         # Metric A (gold denom): grounded / |gold|
#         gold_fact_recall_retrieved = (n_grounded / n_gold) if n_gold > 0 else None

#         # Backwards-compatible column (keep existing name you already output)
#         info_coverage = answer_fact_precision_retrieved

#         # keep column name but new definition: sentence-level ROUGE grounding vs verified facts
#         evidence_nuggets = set(nugget_extractor(verified)) if verified else set()
#         answer_nuggets = nugget_extractor(answer)

#         rougeL = rouge_l_f1(answer, verified) if answer and verified else None

#         if answer and verified:
#             bert_pairs.append((answer, verified))
#             bert_qids.append(qid)

#         per_q_rows.append(
#             {
#                 "model": model,
#                 "strategy": "hierarchy_agents",
#                 "dataset": dataset,
#                 "complexity": comp,
#                 "qid": qid,
#                 # retrieval
#                 "accuracy@10": agg["accuracy@10"],
#                 "recall@10": agg["recall@10"],
#                 "mrr@10": agg["mrr@10"],
#                 "ndcg@10": agg["ndcg@10"],
#                 "n_subqueries_found": len(by_subq),
#                 "n_subquery_lists_scored": len(subquery_metrics),
#                 "n_relevant": sum(1 for _, r in relmap.items() if is_rel(r)) if relmap else 0,
#                 # generation/evidence (kept names; aligned behavior)
#                 "info_coverage": info_coverage,            # now = ROUGE-grounded coverage vs 
#                 "gold_fact_recall_retrieved": gold_fact_recall_retrieved,          # gold-denom
#                 "answer_fact_precision_retrieved": answer_fact_precision_retrieved, # answer-denom (same as info_coverage)
#                 "quality_rougeL": rougeL,                  # answer vs verified_facts ROUGE-L
#                 "quality_bertscore": None,                 # answer vs verified_facts BERTScore
#                 "n_evidence_nuggets": len(evidence_nuggets),
#                 "n_answer_nuggets": n_answer_sents,        # aligned with eval_rag.py meaning/count
#                 "subquery_metrics_json": json.dumps(subquery_metrics, ensure_ascii=False),
#             }
#         )

#     df = pd.DataFrame(per_q_rows)

#     # --- batched minimal BERTScore ---
#     if bert_pairs:
#         f1s = batched_bertscore_f1(bert_pairs, batch_size=64)
#         qid_to_f1: Dict[str, float] = {}
#         for qid, f1 in zip(bert_qids, f1s):
#             qid_to_f1[qid] = float(f1)
#         if not df.empty:
#             df["quality_bertscore"] = df["qid"].map(qid_to_f1)

#     summary = dict(
#         model=model,
#         strategy="hierarchy_agents",
#         dataset=dataset,
#         n_queries=int(df["qid"].nunique()) if not df.empty else 0,
#     )

#     if not df.empty:
#         for col in ["accuracy@10", "recall@10", "mrr@10", "ndcg@10", "info_coverage", "gold_fact_recall_retrieved", "answer_fact_precision_retrieved", "quality_rougeL", "quality_bertscore"]:
#             summary[col] = float(df[col].dropna().mean()) if col in df.columns else None
#         summary["avg_n_subqueries_found"] = float(df["n_subqueries_found"].mean())
#         summary["avg_n_evidence_nuggets"] = float(df["n_evidence_nuggets"].mean())
#         summary["avg_n_answer_nuggets"] = float(df["n_answer_nuggets"].mean())

#     return df, summary


# def summarize_by_complexity(per_q_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Complexity-wise mean metrics, grouped by (model, dataset, complexity).
#     """
#     if per_q_df.empty or "complexity" not in per_q_df.columns:
#         return pd.DataFrame()

#     return (
#         per_q_df.groupby(["model", "dataset", "complexity"], dropna=False)
#         .agg(
#             n_queries=("qid", "nunique"),
#             accuracy_at_10=("accuracy@10", "mean"),
#             recall_at_10=("recall@10", "mean"),
#             mrr_at_10=("mrr@10", "mean"),
#             ndcg_at_10=("ndcg@10", "mean"),
#             info_coverage=("info_coverage", "mean"),
#             gold_fact_recall_retrieved=("gold_fact_recall_retrieved", "mean"),
#             answer_fact_precision_retrieved=("answer_fact_precision_retrieved", "mean"),
#             quality_rougeL=("quality_rougeL", "mean"),
#             quality_bertscore=("quality_bertscore", "mean"),
#             avg_n_subqueries_found=("n_subqueries_found", "mean"),
#             avg_n_evidence_nuggets=("n_evidence_nuggets", "mean"),
#             avg_n_answer_nuggets=("n_answer_nuggets", "mean"),
#         )
#         .reset_index()
#     )


# # -------------------------
# # Main
# # -------------------------
# def main():
#     parser = argparse.ArgumentParser(description="Evaluate hierarchy_agents on any dataset (file name contains 'hierarchy')")
#     parser.add_argument("--inputs", required=True, help="Directory containing JSONL result files")
#     parser.add_argument("--out", required=True, help="Output CSV file for summary metrics")
#     parser.add_argument("--detailed-out", help="Optional output CSV for per-query metrics")
#     parser.add_argument("--by-complexity-out", help="Optional output CSV for complexity-wise metrics")
#     parser.add_argument("--years", default="2019,2020,2021",
#                         help="TREC DL years to include (only applies to trecdl), e.g., 2020 or 2019,2020")
#     args = parser.parse_args()

#     logging.basicConfig(level=logging.INFO)

#     years = [y.strip() for y in args.years.split(",") if y.strip()]
#     allowed_trecdl = {f"trecdl{y}" for y in years}  # only used for trecdl keys

#     input_dir = Path(args.inputs)
#     if not input_dir.exists():
#         raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

#     t0 = time.perf_counter()

#     # 1) Collect relevant files: must contain 'hierarchy' in filename (any dataset)
#     files = []
#     for f in input_dir.glob("*.jsonl"):
#         name = f.name.lower()
#         if "hierarchy_agents" in name or "hierarchy" in name:
#             files.append(f)

#     if not files:
#         raise RuntimeError("No files matched: '*.jsonl' containing 'hierarchy' in the filename.")

#     # 2) Group by (model, dataset_key)
#     grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

#     for f in files:
#         rows = load_jsonl(f)

#         # Determine model name cheaply from filename prefix (everything before '__')
#         stem = f.stem
#         model = stem.split("__")[0] if "__" in stem else stem

#         for r in rows:
#             # strict strategy filter
#             strat = r.get("strategy")
#             if strat and strat != "hierarchy_agents":
#                 continue

#             # determine dataset key from record metadata first, else filename
#             ds_key = extract_dataset_key(r)
#             if ds_key is None:
#                 ds_key = normalize_dataset_key(f.name)

#             if ds_key is None:
#                 continue

#             # apply years filter only for trecdl
#             if ds_key.startswith("trecdl") and ds_key not in allowed_trecdl:
#                 continue

#             grouped[(model, ds_key)].append(r)

#     if not grouped:
#         raise RuntimeError("No hierarchy_agents records found after filtering by strategy/years.")

#     # 3) Cache qrels per dataset_key
#     qrels_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
#     complexity_cache: Dict[str, Dict[str, str]] = {}

#     summaries: List[Dict[str, Any]] = []
#     detailed_frames: List[pd.DataFrame] = []

#     for (model, ds_key), records in grouped.items():
#         logger.info(f"Evaluating: model={model} dataset={ds_key} n_rows={len(records)}")

#         if ds_key not in qrels_cache:
#             bundle = load_bundle_from_dataset_key(ds_key)
#             qrels_cache[ds_key] = make_qrels_map(bundle.qrels_df)
#             complexity_cache[ds_key] = make_complexity_map(bundle.queries_df)

#         per_q_df, summ = evaluate_hierarchy(
#             model=model,
#             dataset=ds_key,
#             records=records,
#             qrels_map=qrels_cache[ds_key],
#             complexity_map=complexity_cache.get(ds_key, {}),
#         )
#         summaries.append(summ)
#         if not per_q_df.empty:
#             detailed_frames.append(per_q_df)

#     # 4) Save outputs
#     out_path = Path(args.out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     summary_df = pd.DataFrame(summaries)
#     summary_df.to_csv(out_path, index=False)
#     logger.info(f"Saved summary to {out_path}")

#     # Build detailed_df once (needed for complexity-wise output)
#     detailed_df = pd.concat(detailed_frames, ignore_index=True) if detailed_frames else pd.DataFrame()

#     if args.detailed_out:
#         detailed_path = Path(args.detailed_out)
#         detailed_path.parent.mkdir(parents=True, exist_ok=True)
#         detailed_df.to_csv(detailed_path, index=False)
#         logger.info(f"Saved detailed to {detailed_path}")

#     if args.by_complexity_out:
#         byc_df = summarize_by_complexity(detailed_df)
#         byc_path = Path(args.by_complexity_out)
#         byc_path.parent.mkdir(parents=True, exist_ok=True)
#         byc_df.to_csv(byc_path, index=False)
#         logger.info(f"Saved by-complexity to {byc_path}")

#     print("\nHierarchy Evaluation Summary (@10 + generation):")
#     print("=" * 100)
#     print(summary_df.to_string(index=False))

#     if args.by_complexity_out and not detailed_df.empty:
#         byc_df = summarize_by_complexity(detailed_df)
#         if not byc_df.empty:
#             print("\nHierarchy Evaluation By Complexity (@10 + generation):")
#             print("=" * 100)
#             print(byc_df.to_string(index=False))

#     logger.info(f"Total eval time: {time.perf_counter() - t0:.2f}s")


# if __name__ == "__main__":
#     main()


# src/evaluation/eval_hierarchy.py
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
K = 10                  # retrieval metrics @K (per subquery, then max)
TOPK_EVID_PER_SUBQ = 10 # evidence pool: take top K per subquery (3 subq -> up to 30)
REL_THRESHOLD = 1.0     # relevant if rel >= 1 for binary metrics; nDCG uses graded rel.

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
        year = dataset_key.replace("trecdl", "")   # "2019"/"2020"/"2021"
        ds_name = f"trecdl_{year}"
        bundle, _ = dm.load_bundle(ds_name, limit=None)
        return bundle

    if dataset_key == "beir_scidocs":
        bundle, _ = dm.load_bundle("beir", limit=None)
        return bundle

    if dataset_key == "beir_nq":
        bundle, _ = dm.load_bundle("nq", limit=None)
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
    """
    Build qid -> complexity label map.
    Expects complexity to exist in bundle.queries_df.
    """
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
# Extraction helpers (hierarchy agent)
# -------------------------
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
    for k in ["final_answer", "answer", "response", "generation", "generated_answer", "output"]:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def extract_verified_facts(rec: Dict[str, Any]) -> str:
    v = rec.get("verified_facts")
    return v.strip() if isinstance(v, str) and v.strip() else ""


def extract_retrieved_docs_by_subquery(rec: Dict[str, Any]) -> Dict[str, List[Tuple[int, str, str]]]:
    """
    Returns:
      { subquery_str : [(rank, docid, passage), ...] }
    based on rec["retrieved_docs"] where each item has source_subquery.
    """
    docs = rec.get("retrieved_docs")
    if not isinstance(docs, list):
        return {}

    out: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)

    for i, item in enumerate(docs):
        if not isinstance(item, dict):
            continue

        subq = item.get("source_subquery") or item.get("subquery") or "UNKNOWN_SUBQUERY"
        subq = str(subq)

        rank = int(item.get("rank", i + 1))
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
            out[subq].append((rank, docid, txt))

    for subq in list(out.keys()):
        out[subq].sort(key=lambda x: x[0])
    return dict(out)

def merge_dedup_ranking_across_subqueries(
    by_subq: Dict[str, List[Tuple[int, str, str]]]
) -> Tuple[List[str], List[str]]:
    """
    Standard IR merged ranking across subqueries:
      - collect all (rank, docid, text) across subqueries
      - sort by rank
      - dedup by docid (keep earliest/highest-ranked occurrence)

    Returns:
      ranked_docids, ranked_texts (aligned)
    """
    all_items: List[Tuple[int, str, str]] = []
    for items in by_subq.values():
        all_items.extend(items)

    all_items.sort(key=lambda x: x[0])

    seen = set()
    docids: List[str] = []
    texts: List[str] = []
    for _, docid, txt in all_items:
        if not docid:
            continue
        if docid in seen:
            continue
        seen.add(docid)
        docids.append(docid)
        texts.append((txt or "").strip())
    return docids, texts


def flatten_retrieved_evidence_dedup(
    by_subq: Dict[str, List[Tuple[int, str, str]]],
    *,
    k_per_subq: int = TOPK_EVID_PER_SUBQ,
) -> List[Tuple[str, str]]:
    """
    Flatten top-k_per_subq docs per subquery into a single deduped evidence list.

    Returns a list of (docid, passage) in stable order.
    Dedup priority:
      1) docid (if present)
      2) normalized passage text (fallback)
    """
    seen_docids = set()
    seen_texts = set()
    out: List[Tuple[str, str]] = []

    for subq, items in by_subq.items():
        for _, docid, txt in items[:k_per_subq]:
            t = (txt or "").strip()
            if not t:
                continue

            did = (docid or "").strip()
            if did:
                if did in seen_docids:
                    continue
                seen_docids.add(did)
            else:
                norm = normalize_text_for_dedup(t)
                if norm in seen_texts:
                    continue
                seen_texts.add(norm)

            out.append((did, t))

    return out


# -------------------------
# Nugget extraction (cheap)
# -------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WS_RE = re.compile(r"\s+")
_PUNCT_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$")


def normalize_text_for_dedup(s: str) -> str:
    s = s.lower().strip()
    s = _WS_RE.sub(" ", s)
    return _PUNCT_EDGE_RE.sub("", s)


def nugget_extractor(text: str) -> List[str]:
    """
    Returns a LIST of normalized sentence-level nuggets.
    - Sentence split
    - Normalization
    - Deduplication
    - Minimum 3 words (instead of 20 characters)
    """
    if not text:
        return []

    parts = _SENT_SPLIT_RE.split(text)

    nuggets: List[str] = []
    seen = set()

    for sent in parts:
        n = normalize_text_for_dedup(sent)

        # Minimum 3 words (more appropriate for nugget-style facts)
        if len(n.split()) >= 3 and n not in seen:
            nuggets.append(n)
            seen.add(n)

    return nuggets



# -------------------------
# GoldFactRecall_retrieved (same as eval_rag.py)
# -------------------------
def gold_fact_recall_retrieved(
    answer: str,
    retrieved_texts: List[str],
    *,
    k: int = K,
) -> Tuple[Optional[float], int, int]:
    """
    GoldFactRecall_retrieved = |N_answer ∩ N_gold| / |N_gold|,
    where N_gold are nuggets extracted from retrieved documents.

    Returns:
      (score or None, n_gold_nuggets, n_answer_nuggets_set)
    """
    if not answer:
        return None, 0, 0

    gold_nuggets = set().union(*(nugget_extractor(t) for t in (retrieved_texts or [])[:k]))
    if not gold_nuggets:
        return None, 0, len(set(nugget_extractor(answer)))

    answer_nuggets = set(nugget_extractor(answer))
    score = len(answer_nuggets & gold_nuggets) / len(gold_nuggets)
    return score, len(gold_nuggets), len(answer_nuggets)


# -------------------------
# ROUGE-L
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
      answer nugget is grounded if max_{evidence_nugget} ROUGE-L(answer_nugget, evidence_nugget) >= thr

    IMPORTANT:
      evidence is nuggetized (not whole documents).
    """
    ans_nuggets = nugget_extractor(answer)
    if not ans_nuggets:
        return None, 0, 0

    ev_nuggets = set().union(
        *(nugget_extractor(t) for t in (evidence_texts or []) if isinstance(t, str) and t.strip())
    )
    if not ev_nuggets:
        return None, 0, len(ans_nuggets)

    ev_list = list(ev_nuggets)
    grounded = 0

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
# Evaluation (Hierarchy agents)
# -------------------------
def evaluate_hierarchy(
    *,
    model: str,
    dataset: str,
    records: List[Dict[str, Any]],
    qrels_map: Dict[str, Dict[str, float]],
    complexity_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    per_q_rows: List[Dict[str, Any]] = []
    complexity_map = complexity_map or {}

    bert_pairs: List[Tuple[str, str]] = []
    bert_spans: Dict[str, Tuple[int, int]] = {}

    for rec in records:
        qid = extract_qid(rec)
        if not qid:
            continue

        comp = complexity_map.get(qid, "unknown")
        answer = extract_answer_text(rec)

        by_subq = extract_retrieved_docs_by_subquery(rec)
        relmap = qrels_map.get(qid, {})

        # -----------------------------------------
        # Retrieval metrics (STANDARD IR):
        # merge + dedup across ALL subqueries
        # -----------------------------------------
        ranked_docids_all, _ranked_texts_all = merge_dedup_ranking_across_subqueries(by_subq)

        if not relmap:
            acc10 = rec10 = mrr10 = ndcg10 = 0.0
        else:
            acc10 = accuracy_hit_at_k(ranked_docids_all, relmap, k=K) if ranked_docids_all else 0.0
            rec10 = recall_at_k(ranked_docids_all, relmap, k=K) if ranked_docids_all else 0.0
            mrr10 = mrr_at_k(ranked_docids_all, relmap, k=K) if ranked_docids_all else 0.0
            ndcg10 = ndcg_at_k(ranked_docids_all, relmap, k=K) if ranked_docids_all else 0.0

        # -----------------------------------------
        # Evidence pool (dedup across subqueries)
        # -----------------------------------------
        evid_items = flatten_retrieved_evidence_dedup(by_subq, k_per_subq=TOPK_EVID_PER_SUBQ)
        evidence_texts = [t for _, t in evid_items]
        evidence_docids = [d for d, _ in evid_items if d]
        E = len(evidence_texts)  # <= 30 after dedup

        # -----------------------------------------
        # Grounding / coverage (must be nugget-vs-nugget
        # in grounding_coverage_rouge(), same as eval_rag.py)
        # -----------------------------------------
        cov, n_grounded, n_answer_nuggets = grounding_coverage_rouge(
            answer,
            evidence_texts,
            thr=0.2,
        )
        info_coverage = cov

        # GoldFactRecall_retrieved (no ROUGE inside)
        gfr, n_gold_nuggets, n_answer_nuggets_set = gold_fact_recall_retrieved(
            answer,
            evidence_texts,
            k=E,  # all deduped evidence docs
        )

        # Quality ROUGE-L: average over evidence docs
        rougeL = (
            sum(rouge_l_f1(answer, t) for t in evidence_texts) / len(evidence_texts)
            if answer and evidence_texts
            else None
        )

        # Queue BERTScore pairs (answer vs each evidence doc)
        if answer and evidence_texts:
            start = len(bert_pairs)
            for t in evidence_texts:
                bert_pairs.append((answer, t))
            bert_spans[qid] = (start, len(bert_pairs))

        evidence_nuggets = set().union(*(nugget_extractor(t) for t in evidence_texts))

        per_q_rows.append(
            {
                "model": model,
                "strategy": "hierarchy_agents",
                "dataset": dataset,
                "complexity": comp,
                "qid": qid,
                # retrieval (STANDARD IR)
                "accuracy@10": acc10,
                "recall@10": rec10,
                "mrr@10": mrr10,
                "ndcg@10": ndcg10,
                "n_subqueries_found": len(by_subq),
                "n_candidates": len(ranked_docids_all),
                "topk_docids": " ".join(ranked_docids_all[:K]),
                "n_relevant": sum(1 for _, r in relmap.items() if is_rel(r)) if relmap else 0,
                # evidence / generation
                "info_coverage": info_coverage,
                "gold_fact_recall_retrieved": gfr,
                "quality_rougeL": rougeL,
                "quality_bertscore": None,  # filled after batched scoring
                "n_evidence_nuggets": len(evidence_nuggets),
                "n_gold_nuggets": n_gold_nuggets,
                "n_answer_nuggets": n_answer_nuggets,
                "n_answer_nuggets_set": n_answer_nuggets_set,
                "n_evidence_docs_dedup": len(evidence_texts),
                "top_evidence_docids": " ".join(evidence_docids),
            }
        )

    df = pd.DataFrame(per_q_rows)

    # --- batched BERTScore ---
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
        strategy="hierarchy_agents",
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
            "quality_rougeL",
            "quality_bertscore",
        ]:
            summary[col] = float(df[col].dropna().mean()) if col in df.columns else None

        summary["avg_n_subqueries_found"] = float(df["n_subqueries_found"].mean())
        summary["avg_n_candidates"] = float(df["n_candidates"].mean())
        summary["avg_n_evidence_nuggets"] = float(df["n_evidence_nuggets"].mean())
        summary["avg_n_gold_nuggets"] = float(df["n_gold_nuggets"].mean())
        summary["avg_n_answer_nuggets"] = float(df["n_answer_nuggets"].mean())
        summary["avg_n_answer_nuggets_set"] = float(df["n_answer_nuggets_set"].mean())
        summary["avg_n_evidence_docs_dedup"] = float(df["n_evidence_docs_dedup"].mean())

    return df, summary


def summarize_by_complexity(per_q_df: pd.DataFrame) -> pd.DataFrame:
    """
    Complexity-wise mean metrics, grouped by (model, dataset, complexity).
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
            info_coverage=("info_coverage", "mean"),
            gold_fact_recall_retrieved=("gold_fact_recall_retrieved", "mean"),
            quality_rougeL=("quality_rougeL", "mean"),
            quality_bertscore=("quality_bertscore", "mean"),
            avg_n_evidence_nuggets=("n_evidence_nuggets", "mean"),
            avg_n_gold_nuggets=("n_gold_nuggets", "mean"),
            avg_n_answer_nuggets=("n_answer_nuggets", "mean"),
            avg_n_answer_nuggets_set=("n_answer_nuggets_set", "mean"),
        )
        .reset_index()
    )


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate hierarchy_agents on any dataset (file name contains 'hierarchy')")
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

    # 1) Collect relevant files
    files = []
    for f in input_dir.glob("*.jsonl"):
        name = f.name.lower()
        if "hierarchy_agents" in name or "hierarchy" in name:
            files.append(f)

    if not files:
        raise RuntimeError("No files matched: '*.jsonl' containing 'hierarchy' in the filename.")

    # 2) Group by (model, dataset_key)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for f in files:
        rows = load_jsonl(f)

        # Determine model name cheaply from filename prefix (everything before '__')
        stem = f.stem
        model = stem.split("__")[0] if "__" in stem else stem

        for r in rows:
            strat = r.get("strategy")
            if strat and strat != "hierarchy_agents":
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
        raise RuntimeError("No hierarchy_agents records found after filtering by strategy/years.")

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

        per_q_df, summ = evaluate_hierarchy(
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

    print("\nHierarchy Evaluation Summary (@10 + generation):")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    if args.by_complexity_out and not detailed_df.empty:
        byc_df = summarize_by_complexity(detailed_df)
        if not byc_df.empty:
            print("\nHierarchy Evaluation By Complexity (@10 + generation):")
            print("=" * 100)
            print(byc_df.to_string(index=False))

    logger.info(f"Total eval time: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
