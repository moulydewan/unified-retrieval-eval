# src/agents/fact_verification_agent.py

import logging
from typing import List, Dict, Any

from src.generation_backend.aws_bedrock_backend import generate_bedrock_model

logger = logging.getLogger(__name__)


class FactVerificationAgent:
    """
    Document-based fact verification.
    Extracts ONLY facts from retrieved documents relevant to the query.
    """

    def __init__(
        self,
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        temperature=0.7,
        max_tokens=512,
    ):
        self.model_id = model_id
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def build_prompt(self, query: str, docs: List[Dict[str, str]]) -> str:
        evidence_text = "\n\n".join(
            [f"[Document {d['docid']}]\n{d['passage']}" for d in docs]
        )

        return (
            "SYSTEM:\n"
            "You are a simple fact verification agent.\n\n"
            "Your task:\n"
            "- Read the user's query.\n"
            "- Read the retrieved documents.\n"
            "- Extract ONLY the factual information in the documents that helps answer the query.\n"
            "- Do NOT add any external knowledge.\n"
            "- Do NOT guess.\n\n"
            "Return:\n"
            "- A concise block of 'verified facts' extracted ONLY from the documents.\n"
            "- If documents do NOT contain enough info, say:\n"
            "  'Insufficient evidence in retrieved documents.'\n\n"
            "USER:\n"
            f"Query:\n{query}\n\n"
            f"Retrieved Documents:\n{evidence_text}\n\n"
            "Return ONLY the verified facts in plain text."
        )

    def verify(self, query: str, docs: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Returns:
          {"text": <verified_facts>, "usage": dict|None}
        """
        prompt = self.build_prompt(query, docs)

        try:
            gen = generate_bedrock_model(
                model_id=self.model_id,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                return_meta=True,
            )
            text = (gen.get("text") or "").strip()
            usage = gen.get("usage")
            return {"text": text, "usage": usage}
        except Exception as e:
            logger.error(f"[FactVerificationAgent] Error: {e}")
            return {
                "text": "Insufficient evidence in retrieved documents.",
                "usage": None,
            }
