import logging
from typing import Any, Dict

from src.generation_backend.aws_bedrock_backend import generate_bedrock_model

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    def __init__(
        self,
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        temperature=0.7,
        max_tokens=4096,
    ):
        self.model_id = model_id
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        self.plan_system_prompt = (
            "You are the Orchestrator Agent in a fixed, linear multi-agent pipeline. "
            "Your job is ONLY to describe the planned steps that downstream agents "
            "will take to answer the user's query.\n\n"
            "IMPORTANT:\n"
            "- You are NOT making decisions.\n"
            "- You are NOT routing.\n"
            "- The pipeline is FIXED and always the same.\n"
            "- Simply output a clear plan of the steps."
        )

        self.synth_system_prompt = (
            "You are the Orchestrator Agent. "
            "Your role is to produce the FINAL ANSWER based ONLY on verified facts "
            "passed to you by the Fact Verification Agent. "
            "Do not rely on external knowledge or hallucinate."
        )

    def plan(self, query: str) -> Dict[str, Any]:
        prompt = (
            "SYSTEM:\n"
            f"{self.plan_system_prompt}\n\n"
            "USER:\n"
            f"Describe the standard plan for answering this query:\n\n{query}\n\n"
            "Plan structure:\n"
            "1. Query Reformulation Agent generates 3 rewritten sub-queries.\n"
            "2. Dense Retrieval Agent retrieves top documents for each sub-query.\n"
            "3. Fact Verification Agent checks correctness across retrieved docs.\n"
            "4. Orchestrator synthesizes final answer.\n\n"
            "Return the plan clearly."
        )

        try:
            gen = generate_bedrock_model(
                model_id=self.model_id,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=min(self.max_tokens, 1024),  # plan doesn't need 4k
                return_meta=True,
            )
            text = (gen.get("text") or "").strip()
            usage = gen.get("usage")
            return {"text": text, "usage": usage}
        except Exception as e:
            logger.error(f"[OrchestratorAgent.plan] Error: {e}")
            return {"text": "Orchestrator failed to generate a plan.", "usage": None}

    def synthesize(self, query: str, verified_facts: str) -> Dict[str, Any]:
        prompt = (
            "SYSTEM:\n"
            f"{self.synth_system_prompt}\n\n"
            "USER:\n"
            f"Original Query:\n{query}\n\n"
            f"Verified Facts:\n{verified_facts}\n\n"
            "Using ONLY the verified facts above, synthesize the final answer.\n"
            "If information is missing or insufficient, explicitly state so."
        )

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
            logger.error(f"[OrchestratorAgent.synthesize] Error: {e}")
            return {"text": "Orchestrator failed to synthesize a final answer.", "usage": None}
