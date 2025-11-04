import logging
from pathlib import Path
import pandas as pd
from jinja2 import Template
from rank_bm25 import BM25Okapi
from src.generation_backend.aws_bedrock_backend import generate_bedrock_model

logger = logging.getLogger(__name__)



# BM25 Retriever
def retrieve_bm25(df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """
    Retrieve top_k passages per query using BM25.
    Returns a DataFrame with [qid, query, docid, passage, rank, bm25_score].
    """
    unique_queries = df[["qid", "query"]].drop_duplicates()
    logger.info(f"Running BM25 retrieval for {len(unique_queries)} queries over {len(df)} passages ...")

    # Build BM25 index
    corpus = df["passage"].astype(str).tolist()
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    retrieval_rows = []
    for _, row in unique_queries.iterrows():
        qid, query = row["qid"], row["query"]
        tokenized_query = query.split()

        # Compute BM25 scores
        scores = bm25.get_scores(tokenized_query)
        top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        for rank, idx in enumerate(top_idxs, start=1):
            retrieval_rows.append({
                "qid": qid,
                "query": query,
                "docid": df.iloc[idx]["docid"],
                "passage": df.iloc[idx]["passage"],
                "rank": rank,
                "bm25_score": float(scores[idx]),
            })

    results = pd.DataFrame(retrieval_rows)
    logger.info(f"Retrieved top-{top_k} documents for {len(unique_queries)} queries.")
    return results


# RAG Synthesis
def load_prompt_template(prompt_path: Path) -> Template:
    """Load a Jinja2 prompt template."""
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
    text = prompt_path.read_text(encoding="utf-8")
    return Template(text)


def rag_synthesis(
    bm25_df: pd.DataFrame,
    model_id: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    top_k: int | None = None,
    prompt_path: Path = Path("prompts/rag_prompt.txt"),
) -> list[dict]:
    """
    Perform retrieval-augmented generation using a Bedrock model.
    Uses a Jinja2 prompt template from the /prompts folder.
    """
    # Default to 20 if not specified
    top_k = top_k or 20

    all_answers = []
    unique_queries = bm25_df[["qid", "query"]].drop_duplicates()
    logger.info(f"Running RAG synthesis for {len(unique_queries)} queries with model: {model_id}")

    # Load prompt template once
    template = load_prompt_template(prompt_path)

    for _, row in unique_queries.iterrows():
        qid, query = row["qid"], row["query"]

        # Collect top passages for this query
        docs = (
            bm25_df[bm25_df["qid"] == qid]
            .sort_values("rank")
            .head(top_k)["passage"]
            .tolist()
        )

        # Render the template
        prompt = template.render(query=query, docs=docs, top_k=top_k)

        try:
            answer = generate_bedrock_model(
                model_id=model_id,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"Generation failed for qid={qid}: {e}")
            answer = ""

        all_answers.append({
            "qid": qid,
            "query": query,
            "answer": answer.strip() if isinstance(answer, str) else answer
        })

    logger.info(f"Generated {len(all_answers)} synthesized answers.")
    return all_answers
