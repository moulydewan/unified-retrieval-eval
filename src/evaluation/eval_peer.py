# # src/evaluation/eval_peer.py
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
# K = 10
# REL_THRESHOLD = 1.0  # relevant if rel >= 1 for binary metrics; nDCG uses graded rel.

# # Nugget config (keep same style as eval_hierarchy: char-length threshold)
# MIN_NUGGET_CHARS = 20


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
#     NOTE: DatasetManager.load_bundle() in this repo does NOT accept year=.
#     """
#     dm = DatasetManager()

#     if dataset_key.startswith("trecdl"):
#         year = dataset_key.replace("trecdl", "")   # "2019"/"2020"/"2021"
#         ds_name = f"trecdl_{year}"                 # trecdl_2019, trecdl_2020, trecdl_2021
#         bundle, _ = dm.load_bundle(ds_name, limit=None)
#         return bundle

#     if dataset_key == "beir_scidocs":
#         bundle, _ = dm.load_bundle("beir", limit=None)
#         return bundle

#     if dataset_key == "beir_nq":
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
# # Extraction helpers (peer_agents)
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


# def extract_ranked_docids_dense_top10(rec: Dict[str, Any], k: int = K) -> List[str]:
#     docs = rec.get("retrieved_docs")
#     if not isinstance(docs, list):
#         return []

#     # If rank exists, sort by rank; else keep order.
#     items = [d for d in docs if isinstance(d, dict)]
#     if any("rank" in d for d in items):
#         try:
#             items.sort(key=lambda x: int(x.get("rank", 10**9)))
#         except Exception:
#             pass

#     ranked: List[str] = []
#     for d in items:
#         docid = d.get("docid") or d.get("doc_id") or d.get("docno") or d.get("pid") or d.get("id")
#         if docid is None:
#             continue
#         ranked.append(str(docid))
#         if len(ranked) >= k:
#             break
#     return ranked


# def extract_judge_block(rec: Dict[str, Any]) -> str:
#     v = rec.get("final_answer")
#     return v.strip() if isinstance(v, str) and v.strip() else ""


# # Winner + final-synth parsing
# _WIN_RE = re.compile(r"Winner\s+Selected\s+by\s+Judge:\s*(.+?)\s*(?:\n|$)", re.IGNORECASE)
# _FINAL_RE = re.compile(
#     r"---\s*FINAL OUTPUT\s*\(Judge-Synthesized Answer\)\s*---\s*(.*)$",
#     re.IGNORECASE | re.DOTALL,
# )


# def extract_winner_label(judge_block: str) -> str:
#     if not judge_block:
#         return ""
#     m = _WIN_RE.search(judge_block)
#     return m.group(1).strip() if m else ""


# def extract_final_synth_answer(judge_block: str) -> str:
#     if not judge_block:
#         return ""
#     m = _FINAL_RE.search(judge_block)
#     return m.group(1).strip() if m else ""


# def winner_answer_text(rec: Dict[str, Any], winner_label: str) -> str:
#     s = (winner_label or "").lower().replace("_", " ")
#     if "analytical" in s:
#         return (rec.get("analytical_answer") or "").strip()
#     if "exploratory" in s:
#         return (rec.get("exploratory_answer") or "").strip()
#     if "verification" in s:
#         return (rec.get("verification_answer") or "").strip()
#     return ""


# # -------------------------
# # Nugget extraction + info coverage (winner -> final)
# # -------------------------
# _SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# _WS_RE = re.compile(r"\s+")
# _PUNCT_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$")


# def normalize_nugget(s: str) -> str:
#     s = s.lower().strip()
#     s = _WS_RE.sub(" ", s)
#     return _PUNCT_EDGE_RE.sub("", s)


# def nugget_extractor(text: str) -> set[str]:
#     if not text:
#         return set()
#     nuggets = set()
#     for sent in _SENT_SPLIT_RE.split(text):
#         n = normalize_nugget(sent)
#         if len(n) >= MIN_NUGGET_CHARS:
#             nuggets.add(n)
#     return nuggets

# def nugget_extractor_list(text: str) -> List[str]:
#     """
#     Aligned with eval_hierarchy.py: returns a LIST of normalized sentence nuggets
#     (deduped, len>=MIN_NUGGET_CHARS).
#     """
#     if not text:
#         return []
#     parts = _SENT_SPLIT_RE.split(text)
#     nuggets: List[str] = []
#     seen = set()
#     for sent in parts:
#         n = normalize_nugget(sent)
#         if len(n) >= MIN_NUGGET_CHARS and n not in seen:
#             nuggets.append(n)
#             seen.add(n)
#     return nuggets



# def nugget_recall(final_text: str, winner_text: str) -> Optional[float]:
#     """
#     coverage = | nuggets(winner) ∩ nuggets(final) | / | nuggets(winner) |
#     """
#     w = nugget_extractor(winner_text)
#     if not w:
#         return None
#     f = nugget_extractor(final_text)
#     return len(w & f) / len(w)


# # -------------------------
# # ROUGE-L (token LCS)
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
#     thr: float = 0.3,
# ) -> Tuple[Optional[float], int, int]:
#     """
#     Same definition as eval_hierarchy.py (and eval_rag.py).

#     Returns:
#       coverage (0..1 or None), grounded_count, total_answer_sents

#     Rule:
#       answer sentence is grounded if max_{ev in evidence} ROUGE-L(sentence, ev) >= thr
#     """
#     ans_sents = nugget_extractor_list(answer)  # List[str]
#     if not ans_sents:
#         return None, 0, 0

#     ev = [t for t in (evidence_texts or []) if isinstance(t, str) and t.strip()]

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
# # Evaluation (peer_agents) + complexity
# # -------------------------
# def evaluate_peer(
#     *,
#     model: str,
#     dataset: str,
#     records: List[Dict[str, Any]],
#     qrels_map: Dict[str, Dict[str, float]],
#     complexity_map: Dict[str, str],  # qid -> complexity (from bundle.queries_df)
# ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
#     per_q_rows: List[Dict[str, Any]] = []

#     bert_pairs: List[Tuple[str, str]] = []
#     bert_qids: List[str] = []

#     for rec in records:
#         qid = extract_qid(rec)
#         if not qid:
#             continue

#         complexity = complexity_map.get(str(qid))

#         # dense retrieval top-10
#         ranked_docids = extract_ranked_docids_dense_top10(rec, k=K)
#         relmap = qrels_map.get(qid, {})

#         if not ranked_docids or not relmap:
#             acc10 = rec10 = mrr10 = ndcg10 = 0.0
#         else:
#             acc10 = accuracy_hit_at_k(ranked_docids, relmap, k=K)
#             rec10 = recall_at_k(ranked_docids, relmap, k=K)
#             mrr10 = mrr_at_k(ranked_docids, relmap, k=K)
#             ndcg10 = ndcg_at_k(ranked_docids, relmap, k=K)

#         # judge transcript parsing
#         judge_block = extract_judge_block(rec)
#         winner_label = extract_winner_label(judge_block)
#         final_synth = extract_final_synth_answer(judge_block)

#         # winner reference answer
#         winner_text = winner_answer_text(rec, winner_label)

#         # ROUGE-grounded info coverage (winner -> final), aligned with eval_hierarchy.py
#         winner_sents = nugget_extractor_list(winner_text) if winner_text else []
#         cov, _, _ = grounding_coverage_rouge(
#             final_synth,
#             winner_sents,
#             thr=0.3,
#         )
#         coverage = cov


#         # ROUGE / BERT (final -> winner)
#         rougeL = rouge_l_f1(final_synth, winner_text) if final_synth and winner_text else None
#         if final_synth and winner_text:
#             bert_pairs.append((final_synth, winner_text))
#             bert_qids.append(qid)

#         per_q_rows.append(
#             {
#                 "model": model,
#                 "strategy": "peer_agents",
#                 "dataset": dataset,
#                 "qid": qid,
#                 "complexity": complexity,
#                 "accuracy@10": acc10,
#                 "recall@10": rec10,
#                 "mrr@10": mrr10,
#                 "ndcg@10": ndcg10,
#                 "coverage": coverage,
#                 "quality_rougeL": rougeL,
#                 "quality_bertscore": None,
#             }
#         )

#     df = pd.DataFrame(per_q_rows)

#     # batched BERTScore
#     if bert_pairs:
#         f1s = batched_bertscore_f1(bert_pairs, batch_size=64)
#         qid_to_f1: Dict[str, float] = {}
#         for qid, f1 in zip(bert_qids, f1s):
#             qid_to_f1[qid] = float(f1)
#         if not df.empty:
#             df["quality_bertscore"] = df["qid"].map(qid_to_f1)

#     # overall summary row
#     summary = dict(
#         model=model,
#         strategy="peer_agents",
#         dataset=dataset,
#         complexity="all",
#         n_queries=int(df["qid"].nunique()) if not df.empty else 0,
#     )

#     if not df.empty:
#         for col in ["accuracy@10", "recall@10", "mrr@10", "ndcg@10", "coverage", "quality_rougeL", "quality_bertscore"]:
#             summary[col] = float(df[col].dropna().mean()) if col in df.columns else None

#     return df, summary


# # -------------------------
# # Main
# # -------------------------
# def main():
#     parser = argparse.ArgumentParser(
#         description="Evaluate peer_agents on any dataset (dense@10 + winner-vs-final quality) with complexity breakdown"
#     )
#     parser.add_argument("--inputs", required=True, help="Directory containing JSONL result files")
#     parser.add_argument("--out", required=True, help="Output CSV file for summary metrics")
#     parser.add_argument("--detailed-out", help="Optional output CSV for per-query metrics")
#     parser.add_argument("--by-complexity-out", help="Optional output CSV containing ONLY per-complexity summary rows (excludes complexity=all).",)
#     parser.add_argument("--years", default="2019,2020,2021", help="TREC DL years to include (only applies to trecdl), e.g., 2020 or 2019,2020",)
#     args = parser.parse_args()

#     logging.basicConfig(level=logging.INFO)

#     years = [y.strip() for y in args.years.split(",") if y.strip()]
#     allowed_trecdl = {f"trecdl{y}" for y in years}

#     input_dir = Path(args.inputs)
#     if not input_dir.exists():
#         raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

#     t0 = time.perf_counter()

#     # 1) Collect relevant files: must contain 'peer' in filename
#     files = []
#     for f in input_dir.glob("*.jsonl"):
#         name = f.name.lower()
#         if "peer_agents" in name or "peer" in name:
#             files.append(f)

#     if not files:
#         raise RuntimeError("No files matched: '*.jsonl' containing 'peer' in the filename.")

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
#             if strat and strat != "peer_agents":
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
#         raise RuntimeError("No peer_agents records found after filtering by strategy/years.")

#     # 3) Cache qrels + complexity per dataset_key
#     qrels_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
#     complexity_cache: Dict[str, Dict[str, str]] = {}  # ds_key -> (qid -> complexity)

#     summaries: List[Dict[str, Any]] = []
#     detailed_frames: List[pd.DataFrame] = []

#     metric_cols = ["accuracy@10", "recall@10", "mrr@10", "ndcg@10", "coverage", "quality_rougeL", "quality_bertscore"]

#     for (model, ds_key), records in grouped.items():
#         logger.info(f"Evaluating: model={model} dataset={ds_key} n_rows={len(records)}")

#         if ds_key not in qrels_cache or ds_key not in complexity_cache:
#             bundle = load_bundle_from_dataset_key(ds_key)
#             qrels_cache[ds_key] = make_qrels_map(bundle.qrels_df)

#             qdf = bundle.queries_df
#             if "qid" in qdf.columns and "complexity" in qdf.columns:
#                 complexity_cache[ds_key] = {
#                     str(row["qid"]): str(row["complexity"])
#                     for _, row in qdf[["qid", "complexity"]].dropna().iterrows()
#                 }
#             else:
#                 complexity_cache[ds_key] = {}

#         per_q_df, overall_summ = evaluate_peer(
#             model=model,
#             dataset=ds_key,
#             records=records,
#             qrels_map=qrels_cache[ds_key],
#             complexity_map=complexity_cache[ds_key],
#         )
#         summaries.append(overall_summ)

#         # per-complexity summary rows
#         if not per_q_df.empty and "complexity" in per_q_df.columns:
#             for comp, g in per_q_df.dropna(subset=["complexity"]).groupby("complexity"):
#                 row = {
#                     "model": model,
#                     "strategy": "peer_agents",
#                     "dataset": ds_key,
#                     "complexity": str(comp),
#                     "n_queries": int(g["qid"].nunique()),
#                 }
#                 for c in metric_cols:
#                     row[c] = float(g[c].dropna().mean()) if c in g.columns else None
#                 summaries.append(row)

#         if not per_q_df.empty:
#             detailed_frames.append(per_q_df)

#     # 4) Save outputs
#     out_path = Path(args.out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     summary_df = pd.DataFrame(summaries)

#     base_cols = ["model", "strategy", "dataset", "complexity", "n_queries"]
#     for c in base_cols + metric_cols:
#         if c not in summary_df.columns:
#             summary_df[c] = None

#     summary_df = summary_df[base_cols + metric_cols]
#     summary_df.to_csv(out_path, index=False)
#     logger.info(f"Saved summary to {out_path}")

#     if args.by_complexity_out:
#         byc_path = Path(args.by_complexity_out)
#         byc_path.parent.mkdir(parents=True, exist_ok=True)

#         # keep only per-complexity rows (exclude the overall 'all' row)
#         byc_df = summary_df[summary_df["complexity"].notna() & (summary_df["complexity"] != "all")].copy()
#         byc_df.to_csv(byc_path, index=False)
#         logger.info(f"Saved by-complexity summary to {byc_path}")

#     if args.detailed_out:
#         detailed_path = Path(args.detailed_out)
#         detailed_path.parent.mkdir(parents=True, exist_ok=True)
#         detailed_df = pd.concat(detailed_frames, ignore_index=True) if detailed_frames else pd.DataFrame()
#         detailed_df.to_csv(detailed_path, index=False)
#         logger.info(f"Saved detailed to {detailed_path}")

#     print("\nPeer Agents Evaluation Summary (@10 + winner-vs-final quality):")
#     print("=" * 100)
#     print(summary_df.to_string(index=False))

#     logger.info(f"Total eval time: {time.perf_counter() - t0:.2f}s")


# if __name__ == "__main__":
#     main()


#NEW

# # src/evaluation/eval_peer.py
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
    NOTE: DatasetManager.load_bundle() in this repo does NOT accept year=.
    """
    dm = DatasetManager()

    if dataset_key.startswith("trecdl"):
        year = dataset_key.replace("trecdl", "")   # "2019"/"2020"/"2021"
        ds_name = f"trecdl_{year}"                 # trecdl_2019, trecdl_2020, trecdl_2021
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
# Extraction helpers (peer_agents)
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


def extract_ranked_docids_dense_top10(rec: Dict[str, Any], k: int = K) -> List[str]:
    docs = rec.get("retrieved_docs")
    if not isinstance(docs, list):
        return []

    # If rank exists, sort by rank; else keep order.
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

# -------------------------
# Winner selection + combined answer text (peer_agents)
# -------------------------
_WINNER_RE = re.compile(
    r"winner\s*selected\s*by\s*judge\s*:\s*(exploratory|analytical|verification)\s*agent",
    re.IGNORECASE,
)

def extract_winner_agent(rec: Dict[str, Any]) -> Optional[str]:
    """
    Returns: "exploratory" | "analytical" | "verification" | None
    Parses rec["final_answer"] for the winner line.
    """
    txt = rec.get("final_answer") or ""
    m = _WINNER_RE.search(txt)
    return m.group(1).lower() if m else None


def strip_analytical_reasoning(text: str) -> str:
    """
    analytical_answer contains 'Reasoning chain: ... Conclusion: ...'
    For grounding, keep only the 'Conclusion' part when present.
    """
    if not text:
        return ""
    t = text.strip()

    m = re.search(r"\bConclusion\s*:\s*", t, flags=re.IGNORECASE)
    if m:
        return t[m.end():].strip()

    # fallback: just remove the header if present
    t = re.sub(r"^\s*Reasoning\s*chain\s*:\s*", "", t, flags=re.IGNORECASE).strip()
    return t


def select_winner_answer(rec: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (winner_answer_text, winner_source)
    winner_source: "winner:analytical" | "winner:exploratory" | "winner:verification" | "winner:none"
    """
    winner = extract_winner_agent(rec)

    if winner == "analytical":
        a = strip_analytical_reasoning((rec.get("analytical_answer") or "").strip())
        return a, "winner:analytical"

    if winner == "exploratory":
        a = (rec.get("exploratory_answer") or "").strip()
        return a, "winner:exploratory"

    if winner == "verification":
        a = (rec.get("verification_answer") or "").strip()
        return a, "winner:verification"

    return "", "winner:none"

def extract_final_output_only(final_answer: str) -> str:
    """
    Keep only the judge "FINAL OUTPUT" content.
    If not found, return the whole string.
    """
    if not final_answer:
        return ""

    # Match the header line and capture everything after it
    m = re.search(
        r"---\s*FINAL\s*OUTPUT.*?---\s*[\r\n]+",
        final_answer,
        flags=re.IGNORECASE,
    )
    if not m:
        return final_answer.strip()

    return final_answer[m.end():].strip()



def build_combined_answer_text(rec: Dict[str, Any]) -> Tuple[str, str]:
    """
    Combined answer = (winner answer) + (final answer).
    This matches your request: nuggets pool from BOTH.
    Returns (combined_text, source_str)
    """
    final_txt = extract_final_output_only((rec.get("final_answer") or "").strip())
    winner_txt, winner_source = select_winner_answer(rec)

    # avoid double counting if identical
    if winner_txt and final_txt and winner_txt.strip() == final_txt.strip():
        return final_txt, f"{winner_source}+final(dedup)"

    parts = []
    if winner_txt:
        parts.append(winner_txt.strip())
    if final_txt:
        parts.append(final_txt.strip())

    return "\n\n".join(parts).strip(), f"{winner_source}+final"


# -------------------------
# Nugget extraction (>=3 words)
# -------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WS_RE = re.compile(r"\s+")
_PUNCT_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$")


def normalize_nugget(s: str) -> str:
    s = s.lower().strip()
    s = _WS_RE.sub(" ", s)
    return _PUNCT_EDGE_RE.sub("", s)


def nugget_extractor(text: str) -> List[str]:
    if not text:
        return []

    nuggets: List[str] = []
    seen = set()

    for sent in _SENT_SPLIT_RE.split(text):
        n = normalize_nugget(sent)
        if len(n.split()) >= 3 and n not in seen:
            nuggets.append(n)
            seen.add(n)

    return nuggets


# -------------------------
# ROUGE-L (token LCS)
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

# -------------------------
# Grounding Coverage (answer nuggets vs evidence nuggets; ROUGE>=0.2)
# -------------------------
def grounding_coverage_rouge(answer: str, evidence_texts: List[str], *, thr: float = 0.2):
    ans_nuggets = nugget_extractor(answer)
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
# GoldFactRecall_retrieved (answer nuggets vs evidence nuggets; exact match)
# -------------------------
def gold_fact_recall_retrieved(
    answer: str,
    retrieved_texts: List[str],
    *,
    k: int = K,
) -> Tuple[Optional[float], int, int]:
    if not answer:
        return None, 0, 0

    texts = (retrieved_texts or [])[:k]
    if not texts:
        return None, 0, len(nugget_extractor(answer))

    gold_nuggets = set()
    for t in texts:
        gold_nuggets.update(nugget_extractor(t))

    answer_nuggets = set(nugget_extractor(answer))

    if not gold_nuggets:
        return None, 0, len(answer_nuggets)

    return (
        len(answer_nuggets & gold_nuggets) / len(gold_nuggets),
        len(gold_nuggets),
        len(answer_nuggets),
    )


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
# Evaluation (peer_agents) + complexity
# -------------------------
# -------------------------
# Evaluation (peer_agents) + complexity
# -------------------------
def evaluate_peer(
    *,
    model: str,
    dataset: str,
    records: List[Dict[str, Any]],
    qrels_map: Dict[str, Dict[str, float]],
    complexity_map: Dict[str, str],  # qid -> complexity (from bundle.queries_df)
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    per_q_rows: List[Dict[str, Any]] = []

    bert_pairs: List[Tuple[str, str]] = []
    bert_counts: Dict[str, int] = {}
    bert_order: List[str] = []  # qid order aligned with appends

    for rec in records:
        qid = extract_qid(rec)
        if not qid:
            continue

        complexity = complexity_map.get(str(qid))

        # dense retrieval top-10
        ranked_docids = extract_ranked_docids_dense_top10(rec, k=K)
        relmap = qrels_map.get(qid, {})

        if not ranked_docids or not relmap:
            acc10 = rec10 = mrr10 = ndcg10 = 0.0
        else:
            acc10 = accuracy_hit_at_k(ranked_docids, relmap, k=K)
            rec10 = recall_at_k(ranked_docids, relmap, k=K)
            mrr10 = mrr_at_k(ranked_docids, relmap, k=K)
            ndcg10 = ndcg_at_k(ranked_docids, relmap, k=K)

        docs = rec.get("retrieved_docs")
        docs = docs if isinstance(docs, list) else []
        evidence_texts = [(d.get("passage") or "") for d in docs[:K] if isinstance(d, dict)]

        answer, answer_source = build_combined_answer_text(rec)

        # Coverage: answer nuggets vs evidence texts (ROUGE>=0.2)
        coverage, _, _ = grounding_coverage_rouge(
            answer,
            evidence_texts,
            thr=0.2,
        )

        # GoldFactRecall_retrieved: exact match nuggets
        gfr, _, _ = gold_fact_recall_retrieved(
            answer,
            evidence_texts,
            k=K,
        )

        # Quality ROUGE-L: mean(answer vs each retrieved doc)
        rougeL = (
            sum(rouge_l_f1(answer, t) for t in evidence_texts) / len(evidence_texts)
            if answer and evidence_texts
            else None
        )

        # Queue BERTScore: mean(answer vs each retrieved doc)
        if answer and evidence_texts:
            bert_order.append(qid)
            bert_counts[qid] = bert_counts.get(qid, 0) + len(evidence_texts)
            for t in evidence_texts:
                bert_pairs.append((answer, t))

        per_q_rows.append(
            {
                "model": model,
                "strategy": "peer_agents",
                "dataset": dataset,
                "qid": qid,
                "complexity": complexity,
                "accuracy@10": acc10,
                "recall@10": rec10,
                "mrr@10": mrr10,
                "ndcg@10": ndcg10,
                "coverage": coverage,
                "gold_fact_recall_retrieved": gfr,
                "quality_rougeL": rougeL,
                "quality_bertscore": None,
                "answer_source": answer_source,
            }
        )

    # Fill BERTScore (average per qid across its K docs)
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

    # overall summary row
    summary = dict(
        model=model,
        strategy="peer_agents",
        dataset=dataset,
        complexity="all",
        n_queries=int(df["qid"].nunique()) if not df.empty else 0,
    )

    if not df.empty:
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
            summary[col] = float(df[col].dropna().mean()) if col in df.columns else None

    return df, summary


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate peer_agents on any dataset (dense@10 + answer-vs-retrieved grounding/quality) with complexity breakdown"
    )
    parser.add_argument("--inputs", required=True, help="Directory containing JSONL result files")
    parser.add_argument("--out", required=True, help="Output CSV file for summary metrics")
    parser.add_argument("--detailed-out", help="Optional output CSV for per-query metrics")
    parser.add_argument(
        "--by-complexity-out",
        help="Optional output CSV containing ONLY per-complexity summary rows (excludes complexity=all).",
    )
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

    # 1) Collect relevant files: must contain 'peer' in filename
    files = []
    for f in input_dir.glob("*.jsonl"):
        name = f.name.lower()
        if "peer_agents" in name or "peer" in name:
            files.append(f)

    if not files:
        raise RuntimeError("No files matched: '*.jsonl' containing 'peer' in the filename.")

    # 2) Group by (model, dataset_key)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for f in files:
        rows = load_jsonl(f)

        # Determine model name cheaply from filename prefix (everything before '__')
        stem = f.stem
        model = stem.split("__")[0] if "__" in stem else stem

        for r in rows:
            # strict strategy filter
            strat = r.get("strategy")
            if strat and strat != "peer_agents":
                continue

            # determine dataset key from record metadata first, else filename
            ds_key = extract_dataset_key(r)
            if ds_key is None:
                ds_key = normalize_dataset_key(f.name)

            if ds_key is None:
                continue

            # apply years filter only for trecdl
            if ds_key.startswith("trecdl") and ds_key not in allowed_trecdl:
                continue

            grouped[(model, ds_key)].append(r)

    if not grouped:
        raise RuntimeError("No peer_agents records found after filtering by strategy/years.")

    # 3) Cache qrels + complexity per dataset_key
    qrels_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
    complexity_cache: Dict[str, Dict[str, str]] = {}  # ds_key -> (qid -> complexity)

    summaries: List[Dict[str, Any]] = []
    detailed_frames: List[pd.DataFrame] = []

    metric_cols = [
        "accuracy@10",
        "recall@10",
        "mrr@10",
        "ndcg@10",
        "coverage",
        "gold_fact_recall_retrieved",
        "quality_rougeL",
        "quality_bertscore",
    ]

    for (model, ds_key), records in grouped.items():
        logger.info(f"Evaluating: model={model} dataset={ds_key} n_rows={len(records)}")

        if ds_key not in qrels_cache or ds_key not in complexity_cache:
            bundle = load_bundle_from_dataset_key(ds_key)
            qrels_cache[ds_key] = make_qrels_map(bundle.qrels_df)

            qdf = bundle.queries_df
            if "qid" in qdf.columns and "complexity" in qdf.columns:
                complexity_cache[ds_key] = {
                    str(row["qid"]): str(row["complexity"])
                    for _, row in qdf[["qid", "complexity"]].dropna().iterrows()
                }
            else:
                complexity_cache[ds_key] = {}

        per_q_df, overall_summ = evaluate_peer(
            model=model,
            dataset=ds_key,
            records=records,
            qrels_map=qrels_cache[ds_key],
            complexity_map=complexity_cache[ds_key],
        )
        summaries.append(overall_summ)

        # per-complexity summary rows
        if not per_q_df.empty and "complexity" in per_q_df.columns:
            for comp, g in per_q_df.dropna(subset=["complexity"]).groupby("complexity"):
                row = {
                    "model": model,
                    "strategy": "peer_agents",
                    "dataset": ds_key,
                    "complexity": str(comp),
                    "n_queries": int(g["qid"].nunique()),
                }
                for c in metric_cols:
                    row[c] = float(g[c].dropna().mean()) if c in g.columns else None
                summaries.append(row)

        if not per_q_df.empty:
            detailed_frames.append(per_q_df)

    # 4) Save outputs
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summaries)

    base_cols = ["model", "strategy", "dataset", "complexity", "n_queries"]
    for c in base_cols + metric_cols:
        if c not in summary_df.columns:
            summary_df[c] = None

    summary_df = summary_df[base_cols + metric_cols]
    summary_df.to_csv(out_path, index=False)
    logger.info(f"Saved summary to {out_path}")

    if args.by_complexity_out:
        byc_path = Path(args.by_complexity_out)
        byc_path.parent.mkdir(parents=True, exist_ok=True)

        # keep only per-complexity rows (exclude the overall 'all' row)
        byc_df = summary_df[summary_df["complexity"].notna() & (summary_df["complexity"] != "all")].copy()
        byc_df.to_csv(byc_path, index=False)
        logger.info(f"Saved by-complexity summary to {byc_path}")

    if args.detailed_out:
        detailed_path = Path(args.detailed_out)
        detailed_path.parent.mkdir(parents=True, exist_ok=True)
        detailed_df = pd.concat(detailed_frames, ignore_index=True) if detailed_frames else pd.DataFrame()
        detailed_df.to_csv(detailed_path, index=False)
        logger.info(f"Saved detailed to {detailed_path}")

    print("\nPeer Agents Evaluation Summary (@10 + answer-vs-retrieved grounding/quality):")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    logger.info(f"Total eval time: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()

