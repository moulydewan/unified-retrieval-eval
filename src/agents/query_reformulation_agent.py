# src/agents/query_reformulation_agent.py

import logging
import re
from typing import Any, Dict, List

from src.generation_backend.aws_bedrock_backend import generate_bedrock_model

logger = logging.getLogger(__name__)


class QueryReformulationAgent:
    """
    Query Reformulation Agent.
    - Generates 3 diverse reformulated sub-queries.
    - Deterministic: always returns exactly 3 rewrites.

    Backward compatible:
      - rewrite(query) -> list[str]          (existing callers)
    New:
      - rewrite_with_usage(query) -> {"text": list[str], "usage": dict|None}
    """

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ):
        self.model_id = model_id
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        self.system_prompt = (
            "You are the Query Reformulation Agent. Rewrite the user's query into 3 related sub-queries.\n\n"
            "Rules:\n"
            "- Return exactly 3 subqueries.\n"
            "- Keep the main named entity EXACTLY as written (do not change spelling).\n"
            "- Stay on the same intent.\n"
            "- Each must be short.\n"
            "- Output ONLY a numbered or bulleted list.\n"
            "- No explanations."
        )

        # Matches:
        # 1. xxx
        # 1) xxx
        # - xxx
        # • xxx
        self._pattern = re.compile(r"^\s*(?:\d+[\.\)]|-|•)\s*(.+)$")

    def _extract_subqueries(self, raw_text: str, query: str) -> List[str]:
        subqueries: List[str] = []
        for line in (raw_text or "").split("\n"):
            match = self._pattern.match(line.strip())
            if match:
                cleaned = match.group(1).strip()
                if cleaned:
                    subqueries.append(cleaned)

        # Guarantee exactly 3
        if len(subqueries) < 3:
            while len(subqueries) < 3:
                subqueries.append(query)
        elif len(subqueries) > 3:
            subqueries = subqueries[:3]

        return subqueries

    def rewrite_with_usage(self, query: str) -> Dict[str, Any]:
        """
        Returns:
          {"text": list[str], "usage": dict|None}
        """
        prompt = (
            "SYSTEM:\n"
            f"{self.system_prompt}\n\n"
            "USER:\n"
            "Rewrite the following query into 3 related sub-queries:\n\n"
            f"{query}\n"
        )

        try:
            gen = generate_bedrock_model(
                model_id=self.model_id,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                return_meta=True,
            )
            raw_text = (gen.get("text") or "").strip()
            usage = gen.get("usage")
        except Exception as e:
            logger.error(f"[QueryReformulationAgent] Model error: {e}")
            return {"text": [query, query, query], "usage": None}

        subqueries = self._extract_subqueries(raw_text, query)
        logger.info(f"[QueryReformulationAgent] Subqueries: {subqueries}")
        return {"text": subqueries, "usage": usage}

    def rewrite(self, query: str) -> List[str]:
        """
        Backward compatible: returns only list[str] (exactly 3).
        """
        return self.rewrite_with_usage(query).get("text", [query, query, query])
