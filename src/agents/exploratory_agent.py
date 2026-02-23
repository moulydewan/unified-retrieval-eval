import logging
from typing import Any, Dict, List

from src.generation_backend.aws_bedrock_backend import generate_bedrock_model

logger = logging.getLogger(__name__)


class Exploratory_Agent:
    def __init__(
        self,
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        temperature=0.7,
        max_tokens=512,
    ):
        # store params so we can call Bedrock with return_meta=True
        self.model_id = model_id
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        self.system_prompt = (
            "You are Agent A: an Exploratory Search Agent.\n\n"
            "Your role:\n"
            "- Interpret the user query broadly and consider multiple possible meanings.\n"
            "- Explore diverse angles, contexts, and possibilities.\n"
            "- Synthesize information across all retrieved documents.\n"
            "- Provide high-level insights, thematic patterns, and alternative viewpoints.\n"
            "- Avoid strict fact-checking; focus on wide-ranging interpretation and exploration.\n\n"
            "Your goal is to produce a creative, exploratory synthesis of the answer."
        )

    def build_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        docs_formatted = ""
        for i, d in enumerate(documents, start=1):
            docid = d.get("docid", "")
            passage = d.get("passage", "")
            docs_formatted += f"({i}) docid={docid}\n{passage}\n\n"

        user_prompt = (
            f"USER QUERY:\n{query}\n\n"
            f"DOCUMENTS:\n{docs_formatted}\n"
            "TASK:\n"
            "Generate an Exploratory Search Answer.\n\n"
            "Requirements:\n"
            "- Explore multiple interpretations of the query.\n"
            "- Summarize patterns and themes across documents.\n"
            "- Provide a broad, conceptual, high-level answer.\n"
            "- Do NOT list documents individually; synthesize them.\n\n"
            "Return only the final answer — no analysis steps."
        )

        # Single string prompt (system + user), so we can get Bedrock usage reliably
        return (
            "SYSTEM:\n"
            f"{self.system_prompt}\n\n"
            "USER:\n"
            f"{user_prompt}\n"
        )

    def run(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Returns:
          {"text": <answer>, "usage": {"inputTokens":..., "outputTokens":..., "totalTokens":...} }
        """
        logger.info("Agent A generating exploratory synthesis...")

        prompt = self.build_prompt(query, documents)

        try:
            gen = generate_bedrock_model(
                model_id=self.model_id,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                return_meta=True,  # <-- required for usage
            )
            text = (gen.get("text") or "").strip()
            usage = gen.get("usage")  # may be None if backend doesn't provide
            return {"text": text, "usage": usage}
        except Exception as e:
            logger.error(f"[Exploratory_Agent] Error: {e}")
            return {
                "text": "Exploratory agent failed to generate an answer.",
                "usage": None,
            }
