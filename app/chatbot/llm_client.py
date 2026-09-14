import re
from dataclasses import dataclass
from typing import Optional

import httpx

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS

SYSTEM_PROMPT = (
    "You are a programming assistant embedded in a Q&A chatbot. Answer questions about "
    "any programming language or software development concept — syntax, algorithms, "
    "data structures, tooling, best practices, debugging, and so on.\n\n"
    "Rules:\n"
    "- If the question is not about programming or software development, say briefly "
    "that you can only help with programming topics — don't answer it.\n"
    "- Keep the explanation short: 1-4 sentences.\n"
    "- Include one small, runnable code example in a fenced code block when it would "
    "help, using the language the question is about.\n"
    "- Don't pad the answer with disclaimers, greetings, or restating the question."
)

# Matches the first fenced code block in a response, e.g. ```python\ncode\n```
_CODE_BLOCK_PATTERN = re.compile(r"```[a-zA-Z0-9_+-]*\n?(.*?)```", re.DOTALL)


@dataclass
class LLMAnswer:
    answer: str
    code_example: Optional[str]
    error: Optional[str]


def _split_answer_and_code(raw_text: str) -> tuple:
    match = _CODE_BLOCK_PATTERN.search(raw_text)
    if not match:
        return raw_text.strip(), None

    code_example = match.group(1).strip()
    answer = _CODE_BLOCK_PATTERN.sub("", raw_text).strip()
    return answer, (code_example or None)


class LLMClient:
    """
    Thin wrapper around a local Ollama server. Used only when the curated
    Python knowledge base has no confident match, so the chatbot can still
    answer questions about other languages and broader concepts — without
    ever calling a paid or external API.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout_seconds: float = OLLAMA_TIMEOUT_SECONDS,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds

    def generate_answer(self, question: str) -> LLMAnswer:
        try:
            response = httpx.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "system": SYSTEM_PROMPT,
                    "prompt": question,
                    "stream": False,
                    "think": False,
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return LLMAnswer(answer="", code_example=None, error=str(exc))

        raw_text = response.json().get("response", "").strip()
        if not raw_text:
            return LLMAnswer(answer="", code_example=None, error="empty response from model")

        answer, code_example = _split_answer_and_code(raw_text)
        return LLMAnswer(answer=answer, code_example=code_example, error=None)
