import logging
from typing import Any, Dict

from src.generation_backend.aws_bedrock_backend import generate_bedrock_model

logger = logging.getLogger(__name__)


class Judge_Agent:
    def __init__(
        self,
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        temperature=0.2,  # for consistency
        max_tokens=512,
    ):
        self.model_id = model_id
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        self.system_prompt = (
            "You are the Judge Agent in a multi-agent retrieval system. You must produce your output in the exact template below.\n\n"
            "=====================================\n"
            "JUDGE OUTPUT TEMPLATE\n"
            "=====================================\n\n"
            "--- JUDGE AGENT — ROUND 1 REVIEW ---\n"
            "Judge Agent Evaluation:\n"
            "- Exploratory Agent: short critique\n"
            "- Analytical Agent: short critique\n"
            "- Verification Agent: short critique\n"
            "Judge proceeds to Round 2 for scoring.\n\n"
            "--- JUDGE AGENT — ROUND 2 SELF-DEBATE & SCORING ---\n"
            "Judge internal scoring based on:\n"
            "- Evidence alignment\n"
            "- Explanation quality\n"
            "- Completeness\n"
            "- Accuracy\n\n"
            "Scores (out of 10):\n"
            "- Exploratory Agent: score\n"
            "- Analytical Agent: score\n"
            "- Verification Agent: score\n"
            "Winner Selected by Judge: agent name\n\n"
            "--- FINAL OUTPUT (Judge-Synthesized Answer) ---\n"
            "Provide a concise (3 to 5 sentence) final answer fully supported by evidence.\n\n"
            "=====================================\n"
            "RULES\n"
            "=====================================\n"
            "- Follow the template provided.\n"
            "- DO NOT create new headings.\n"
            "- DO NOT reorder steps.\n"
            "- DO NOT include chain-of-thought.\n"
            "- FINAL ANSWER must be concise and factual.\n"
        )

    def _as_text(self, x) -> str:
        return "" if x is None else str(x).strip()

    def build_prompt(self, query, exploratory_output, analytical_output, verification_output) -> str:
        query = self._as_text(query)
        exploratory_output = self._as_text(exploratory_output)
        analytical_output = self._as_text(analytical_output)
        verification_output = self._as_text(verification_output)

        user_prompt = (
            f"USER QUERY:\n{query}\n\n"
            "===== AGENT ANSWERS =====\n"
            f"Agent A (Exploratory):\n{exploratory_output}\n\n"
            f"Agent B (Analytical):\n{analytical_output}\n\n"
            f"Agent C (Verification):\n{verification_output}\n\n"
            "Follow the output template.\n"
        )

        return (
            "SYSTEM:\n"
            f"{self.system_prompt}\n\n"
            "USER:\n"
            f"{user_prompt}\n"
        )

    def evaluate(self, query, exploratory_output, analytical_output, verification_output) -> Dict[str, Any]:
        """
        Returns:
          {"text": <judge_output>, "usage": <bedrock usage dict>}
        """
        logger.info("Judge Agent evaluating final answer...")

        prompt = self.build_prompt(
            query=query,
            exploratory_output=exploratory_output,
            analytical_output=analytical_output,
            verification_output=verification_output,
        )

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
            logger.error(f"[Judge_Agent] Error: {e}")
            return {"text": "Judge agent failed to generate a final answer.", "usage": None}
