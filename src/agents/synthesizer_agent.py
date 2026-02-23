# src/agents/synthesizer_agent.py
import logging
from typing import Dict, Any, List

from src.generation_backend.aws_bedrock_backend import generate_bedrock_model

logger = logging.getLogger(__name__)


def _truncate(text: str, max_chars: int) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


class SynthesizerAgent:
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
            "You are a Synthesizer Agent.\n"
            "Given the query, passages, and analyzer findings (plain text), produce a clear answer.\n"
            "Rules:\n"
            "- Use ONLY information supported by the passages.\n"
            "- If evidence is missing, say so explicitly.\n"
            "- If evidence conflicts, present the best-supported view and note uncertainty.\n"
            "- Cite docids inline like [docid=XYZ] (use the exact docid strings).\n"
            "- Be concise (3–6 sentences).\n"
            "Return only the answer."
        )

    def run_with_usage(self, query: str, docs: List[Dict[str, Any]], analyzer_output: str) -> Dict[str, Any]:
        analyzer_text = _truncate(analyzer_output, 4000)

        per_doc_chars = 900
        max_total_chars = 9000

        chunks = []
        total = 0
        for d in docs:
            docid = str(d.get("docid", ""))
            passage = _truncate((d.get("passage") or "").strip(), per_doc_chars)
            if not passage:
                continue
            chunk = f"[docid={docid}] {passage}"
            if total + len(chunk) > max_total_chars:
                break
            chunks.append(chunk)
            total += len(chunk) + 2

        doc_blob = "\n\n".join(chunks)

        prompt = (
            "SYSTEM:\n" + self.system_prompt + "\n\n"
            "USER:\n"
            f"QUERY:\n{query}\n\n"
            f"ANALYZER_ANALYSIS:\n{analyzer_text}\n\n"
            f"PASSAGES:\n{doc_blob}\n\n"
            "Write the final synthesized answer."
        )

        try:
            gen = generate_bedrock_model(
                model_id=self.model_id,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                return_meta=True,
            )
            return {"text": (gen.get("text") or "").strip(), "usage": gen.get("usage")}
        except Exception as e:
            logger.warning(f"[SynthesizerAgent] invoke failed: {e}")
            return {"text": "", "usage": None}

    def run(self, query: str, docs: List[Dict[str, Any]], analyzer_output: str) -> str:
        return self.run_with_usage(query, docs, analyzer_output)["text"]
