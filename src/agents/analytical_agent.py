import logging
from typing import Any, Dict, List

from src.generation_backend.aws_bedrock_backend import generate_bedrock_model

logger = logging.getLogger(__name__)


class Analytical_Agent:
    def __init__(
        self,
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        temperature=0.7,
        max_tokens=512,
    ):
        """
        Analytical Agent:
        - Focuses on structured reasoning.
        - Builds step-by-step logic linking evidence to conclusions.
        - Produces crisp, coherent, rational answers.
        """
        self.model_id = model_id
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        self.system_prompt = (
            "You are Agent B: an Analytical Search Agent.\n\n"
            "Your role:\n"
            "- Perform structured, step-by-step reasoning.\n"
            "- Identify relationships, causal links, factual claims, and contradictions.\n"
            "- Evaluate evidence logically using reasoning chains.\n"
            "- Highlight important facts that strongly affect the conclusion.\n"
            "- Produce a precise, coherent, logically justified answer.\n\n"
            "Your output must be:\n"
            "- Analytical rather than exploratory.\n"
            "- Based on reasoning rather than creativity.\n"
            "- A clear logical conclusion supported by reasoning steps.\n"
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
            "Provide an analytical answer grounded in reasoning.\n\n"
            "Requirements:\n"
            "- Use structured reasoning to interpret the documents.\n"
            "- Identify key factual elements relevant to the query.\n"
            "- If contradictions exist, analyze them logically.\n"
            "- Construct a concise reasoning chain.\n"
            "- End with a clear, justified final conclusion.\n\n"
            "Return only the final answer (with reasoning chain included)."
        )

        return (
            "SYSTEM:\n"
            f"{self.system_prompt}\n\n"
            "USER:\n"
            f"{user_prompt}\n"
        )

    def run(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Returns:
          {"text": <answer>, "usage": <bedrock usage dict>}
        """
        logger.info("Agent B generating analytical reasoning...")

        prompt = self.build_prompt(query, documents)

        try:
            gen = generate_bedrock_model(
                model_id=self.model_id,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                return_meta=True,  # <-- required
            )
            text = (gen.get("text") or "").strip()
            usage = gen.get("usage")
            return {"text": text, "usage": usage}
        except Exception as e:
            logger.error(f"[Analytical_Agent] Error: {e}")
            return {
                "text": "Analytical agent failed to generate an answer.",
                "usage": None,
            }
