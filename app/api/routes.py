from fastapi import APIRouter, Request

from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api")

FALLBACK_MESSAGE = (
    "I couldn't find or generate an answer to that. Try asking about Python variables, "
    "data types, operators, conditional statements, loops, functions, "
    "lists/tuples/sets/dictionaries, strings, object-oriented programming, exception "
    "handling, file handling, modules and packages, common interview questions — or any "
    "other programming language or concept."
)


@router.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@router.get("/categories")
def list_categories(request: Request) -> dict:
    knowledge_base = request.app.state.knowledge_base
    return {"categories": knowledge_base.categories()}


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    matcher = request.app.state.matcher
    result = matcher.match(payload.message)

    if result.matched:
        entry = result.entry
        return ChatResponse(
            answer=entry["answer"],
            matched=True,
            confidence=round(result.score, 4),
            category=entry["category"],
            code_example=entry["code_example"] or None,
            matched_question=entry["question"],
            source="knowledge_base",
        )

    # Not in the curated Python KB with enough confidence — ask the local
    # LLM instead of giving up, so other languages and broader concepts are
    # still answerable. If Ollama is unreachable or errors out, fall back to
    # the honest static message rather than a 500.
    llm_client = request.app.state.llm_client
    llm_result = llm_client.generate_answer(payload.message)

    if llm_result.error:
        return ChatResponse(
            answer=FALLBACK_MESSAGE,
            matched=False,
            confidence=round(result.score, 4),
            source="none",
        )

    return ChatResponse(
        answer=llm_result.answer,
        matched=False,
        confidence=round(result.score, 4),
        code_example=llm_result.code_example,
        source="ai",
    )
