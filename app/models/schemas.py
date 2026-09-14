from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be blank")
        return stripped


class ChatResponse(BaseModel):
    answer: str
    matched: bool
    confidence: float
    category: Optional[str] = None
    code_example: Optional[str] = None
    matched_question: Optional[str] = None
    # "knowledge_base": a verified answer from the curated Python KB (matched=True).
    # "ai": generated on the fly by the local LLM for anything outside the KB.
    # "none": neither the KB nor the LLM could answer (e.g. Ollama unreachable).
    source: Literal["knowledge_base", "ai", "none"] = "none"
