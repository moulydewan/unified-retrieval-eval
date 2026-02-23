import logging
from typing import Any, Dict, List

from src.generation_backend.aws_bedrock_backend import generate_bedrock_model

logger = logging.getLogger(__name__)


class Verification_Agent:
    def __init__(
        self,
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        temperature=0.7,
        max_tokens=1024,
    ):
        """
        Verification Agent:
        - Strict, evidence-based reasoning.
        - Cross-checks facts across documents.
        - Flags contradictions, missing evidence, and unsupported claims.
        - Produces a short, precise answer grounded ONLY in the documents.
        """
        self.model_id = model_id
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        self.system_prompt = (
            "You are Agent C: a Verification Agent.\n\n"
            "Your role:\n"
            "- Provide a short, factual answer strictly grounded in the documents.\n"
            "- Verify all claims using evidence from the text.\n"
            "- If information is missing or contradictory, state this clearly.\n"
            "- Do NOT speculate or infer beyond what is supported.\n"
            "- Be concise and evidence-based rather than explanatory.\n\n"
            "Your output must:\n"
            "- Be factual\n"
            "- Cite document numbers when relevant (e.g., 'Doc 2 states...')\n"
            "- Avoid long reasoning chains\n"
            "- Provide a final verified conclusion"
        )

    def build_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        docs_formatted = ""
        for i, d in enumerate(documents, start=1):
            docid = d.get("docid", "")
            passage = d.get("passage", "")
            docs_formatted += f"[Doc {i}] (docid={docid})\n{passage}\n\n"

        user_prompt = (
            f"USER QUERY:\n{query}\n\n"
            f"DOCUMENTS:\n{docs_formatted}\n"
            "TASK:\n"
            "Provide a short, evidence-based answer strictly grounded in the documents.\n\n"
            "Requirements:\n"
            "- Verify claims using document evidence.\n"
            "- Cite Doc numbers when supporting statements.\n"
            "- If information is missing or contradictory, say so.\n"
            "- Avoid speculation.\n"
            "- Keep the output factual and concise.\n\n"
            "Return only the verified answer."
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
        logger.info("Agent C generating verification-based answer...")

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
            usage = gen.get("usage")
            return {"text": text, "usage": usage}
        except Exception as e:
            logger.error(f"[Verification_Agent] Error: {e}")
            return {
                "text": "Verification agent failed to generate an answer.",
                "usage": None,
            }
