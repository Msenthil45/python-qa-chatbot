from fastapi import APIRouter, Request

from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api")

FALLBACK_MESSAGE = (
    "I don't have information about that in my knowledge base yet. Try asking about "
    "Python variables, data types, operators, conditional statements, loops, functions, "
    "lists/tuples/sets/dictionaries, strings, object-oriented programming, exception "
    "handling, file handling, modules and packages, or common interview questions."
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

    if not result.matched:
        return ChatResponse(
            answer=FALLBACK_MESSAGE,
            matched=False,
            confidence=round(result.score, 4),
        )

    entry = result.entry
    return ChatResponse(
        answer=entry["answer"],
        matched=True,
        confidence=round(result.score, 4),
        category=entry["category"],
        code_example=entry["code_example"] or None,
        matched_question=entry["question"],
    )
