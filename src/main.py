import logging
import pandas as pd
from pathlib import Path
from src.datasets.trecdl import TRECDLAdapter
from src.datasets.beir import BEIRSCIDOCSAdapter
from src.datasets.naturalquestion import NaturalQuestionsAdapter
from src.managers.dataset_manager import DatasetManager
from src.utils import setup_logging, save_jsonl, create_run_id, ExperimentTracker
from src.datasets.datasetbundle import IRDatasetBundle


# Setup experiment logging
logger = setup_logging(Path("outputs/logs"), "data")
tracker = ExperimentTracker(Path("outputs"))
SUBSETS_DIR = Path("subsets")


import pandas as pd

from src.datasets.trecdl import TRECDLAdapter
from src.datasets.beir import BEIRSCIDOCSAdapter
from src.datasets.naturalquestion import NaturalQuestionsAdapter
from pathlib import Path



# def summarize_complexity_from_subset(name: str, subset_path: Path):
#     if not subset_path.exists():
#         print(f"\n{name} complexity: (missing) {subset_path}")
#         return

#     df = pd.read_json(subset_path, lines=True)
#     if "complexity" not in df.columns:
#         print(f"\n{name} complexity: no 'complexity' column in {subset_path}")
#         return

#     vc = df["complexity"].value_counts(dropna=False)
#     print(f"\n{name} complexity distribution ({subset_path.name})")
#     print("-" * 70)
#     print(f"n = {len(df)}")
#     print(vc.to_string())



# def get_rel_col(qrels_df: pd.DataFrame) -> str:
#     for c in ["relevance", "rel", "score", "label", "judgment"]:
#         if c in qrels_df.columns:
#             return c
#     raise ValueError(f"No relevance column found. Columns: {list(qrels_df.columns)}")


# def summarize_qrels(name: str, qrels_df: pd.DataFrame):
#     rcol = get_rel_col(qrels_df)
#     vals = pd.to_numeric(qrels_df[rcol], errors="coerce").dropna().astype(float)

#     print(f"\nDataset: {name}")
#     print("-" * 70)
#     print(f"Rel column   : {rcol}")
#     print(f"Total qrels  : {len(vals)}")
#     print(f"Min / Max    : {vals.min()} / {vals.max()}")
#     print(f"Unique values: {sorted(vals.unique().tolist())}")
#     print("Value counts:")
#     print(vals.value_counts().sort_index().to_string())


# def main():
#     # BEIR
#     beir = BEIRSCIDOCSAdapter(data_dir="datasets/beir", split="test")
#     beir_bundle = beir.load_bundle(limit=None)
#     summarize_qrels("beir", beir_bundle.qrels_df)

#     # NQ
#     nq = NaturalQuestionsAdapter(split="dev")
#     nq_bundle = nq.load_bundle(limit=None)
#     summarize_qrels("nq", nq_bundle.qrels_df)

#     # TREC DL years
#     for year in [2019, 2020, 2021]:
#         trec = TRECDLAdapter(year=year, mode="passage")
#         trec_bundle = trec.load_bundle(limit=None)
#         summarize_qrels(f"trecdl{year}", trec_bundle.qrels_df)

#         # Complexity (from saved subset files)
#     summarize_complexity_from_subset("trec_dl_2019", SUBSETS_DIR / "trec_dl_2019_50.jsonl")
#     summarize_complexity_from_subset("trec_dl_2020", SUBSETS_DIR / "trec_dl_2020_50.jsonl")
#     summarize_complexity_from_subset("trec_dl_2021", SUBSETS_DIR / "trec_dl_2021_50.jsonl")
#     summarize_complexity_from_subset("beir_scidocs", SUBSETS_DIR / "beir_scidocs_50.jsonl")
#     summarize_complexity_from_subset("beir_nq", SUBSETS_DIR / "beir_nq_50.jsonl")


# if __name__ == "__main__":
#     main()

#QUERY DISTRIBUTION

# def print_queries_by_complexity(path: Path):
#     df = pd.read_json(path, lines=True)

#     print(f"\n=== {path.name} ===")
#     for label in ["complex", "moderate", "simple"]:
#         sub = df[df["complexity"] == label]
#         print(f"\n[{label.upper()}] (n={len(sub)})")
#         print("-" * 80)
#         for i, q in enumerate(sub["query"], 1):
#             print(f"{i:2d}. {q}")

# BASE = Path("subsets")

# print_queries_by_complexity(BASE / "trec_dl_2019_50.jsonl")
# print_queries_by_complexity(BASE / "trec_dl_2020_50.jsonl")
# print_queries_by_complexity(BASE / "trec_dl_2021_50.jsonl")

#PLOT TABLE 1
# src/main.py
# src/main.py
from collections import Counter
from src.managers.dataset_manager import DatasetManager


DATASETS = [
    "trecdl_2019",
    "trecdl_2020",
    "trecdl_2021",
    "beir",
    "nq",
]


def summarize_bundle(name, bundle):
    qdf = bundle.queries_df
    qrels = bundle.qrels_df
    corpus = getattr(bundle, "corpus_df", None)

    stats = {}

    stats["queries"] = qdf["qid"].nunique()
    stats["qrels"] = len(qrels)
    stats["judged_docs"] = qrels["docid"].nunique()
    stats["full_corpus"] = len(corpus) if corpus is not None else None

    if "complexity" in qdf.columns:
        stats["complexity"] = Counter(qdf["complexity"].dropna())
    else:
        stats["complexity"] = {}

    return stats


def main():
    dm = DatasetManager()

    print("\nDataset Summary")
    print("=" * 80)

    for ds_key in DATASETS:
        bundle, _ = dm.load_bundle(ds_key, limit=None)
        stats = summarize_bundle(ds_key, bundle)

        print(f"\n{ds_key.upper()}")
        print("-" * 40)
        print(f"#Queries        : {stats['queries']}")
        print(f"#Judged docs    : {stats['judged_docs']}")
        print(f"#Qrels          : {stats['qrels']}")

        if stats["full_corpus"] is not None:
            print(f"#Full corpus    : {stats['full_corpus']}")
        else:
            print(f"#Full corpus    : judged-only")

        if stats["complexity"]:
            print("Complexity dist:")
            for k in ["simple", "moderate", "complex"]:
                if k in stats["complexity"]:
                    print(f"  {k:<9}: {stats['complexity'][k]}")
        else:
            print("Complexity dist: N/A")


if __name__ == "__main__":
    main()
