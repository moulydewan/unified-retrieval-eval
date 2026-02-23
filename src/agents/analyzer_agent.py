# src/agents/analyzer_agent.py
import logging
from typing import Dict, Any, List

from src.generation_backend.aws_bedrock_backend import generate_bedrock_model

logger = logging.getLogger(__name__)


class AnalyzerAgent:
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
            "You are an Analyzer Agent.\n"
            "Given a user query and a set of retrieved passages, produce a concise analysis to support answer synthesis.\n\n"
            "Use the following structure, in this order:\n"
            "KEY FACTS:\n"
            "MISSING INFORMATION:\n"
            "CONTRADICTIONS:\n\n"
            "Formatting rules:\n"
            "- Use plain text only (no JSON).\n"
            "- Under each header, use bullet points starting with '- '.\n"
            "- In KEY FACTS and CONTRADICTIONS, cite docid(s) like '(docid=...)'.\n"
            "- If a section has nothing, write '- None'.\n"
            "- Be concise and specific.\n"
        )

    def _fallback(self) -> str:
        return (
            "KEY FACTS:\n- None\n\n"
            "MISSING INFORMATION:\n- None\n\n"
            "CONTRADICTIONS:\n- None"
        )

    def run_with_usage(self, query: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        doc_blob = "\n\n".join(
            [
                f"[docid={d.get('docid')} source={d.get('source','')} rank={d.get('rank','')}] "
                f"{(d.get('passage') or '')}"
                for d in docs
            ]
        )

        prompt = (
            "SYSTEM:\n" + self.system_prompt + "\n\n"
            "USER:\n"
            f"QUERY:\n{query}\n\n"
            f"PASSAGES:\n{doc_blob}\n\n"
            "Return the analysis in the required plain-text format."
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
            if not text:
                text = self._fallback()
            return {"text": text, "usage": usage}
        except Exception as e:
            logger.warning(f"[AnalyzerAgent] invoke failed: {e}")
            return {"text": self._fallback(), "usage": None}

    def run(self, query: str, docs: List[Dict[str, Any]]) -> str:
        # backward-compatible: still returns str
        return self.run_with_usage(query, docs)["text"]
