# src/agents/validator_agent.py
import logging
from typing import Dict, Any, List

from src.generation_backend.aws_bedrock_backend import generate_bedrock_model

logger = logging.getLogger(__name__)


def _truncate(text: str, max_chars: int) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


class ValidatorAgent:
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
            "You are a Validator Agent.\n"
            "Task: Verify the draft answer against the passages.\n\n"
            "Rules:\n"
            "- Do NOT add new facts not present in passages.\n"
            "- Remove or rewrite any unsupported claim.\n"
            "- If evidence is insufficient, explicitly say what cannot be verified.\n"
            "- Cite docids inline like [docid=XYZ] for each key claim.\n"
            "- Keep it concise (3–6 sentences).\n"
            "- Return ONLY the final validated answer (no headings, no analysis)."
        )

    def run_with_usage(self, query: str, docs: List[Dict[str, Any]], draft_answer: str) -> Dict[str, Any]:
        draft_answer = _truncate(draft_answer, 2000)

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
            f"DRAFT_ANSWER:\n{draft_answer}\n\n"
            f"PASSAGES:\n{doc_blob}\n\n"
            "Return the final validated answer."
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
            logger.warning(f"[ValidatorAgent] invoke failed: {e}")
            return {"text": "", "usage": None}

    def run(self, query: str, docs: List[Dict[str, Any]], draft_answer: str) -> str:
        return self.run_with_usage(query, docs, draft_answer)["text"]
