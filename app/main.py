from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.chatbot.hybrid_matcher import HybridMatcher
from app.chatbot.knowledge_base import KnowledgeBase
from app.chatbot.matcher import QuestionMatcher
from app.chatbot.semantic_matcher import SemanticMatcher
from app.config import FRONTEND_DIR, KNOWLEDGE_BASE_PATH

app = FastAPI(
    title="Python Q&A Chatbot",
    description="A knowledge-base-driven chatbot that answers Python programming questions.",
    version="0.1.0",
)

# Built once at import time rather than in an on_event("startup") handler —
# this keeps the app fully initialized as soon as it's imported (no lifespan
# dance needed to exercise it from tests). Loading the sentence-embedding
# model is the slow part of startup (a couple of seconds once cached
# locally, longer on the very first run while it downloads); everything
# else here is fast.
app.state.knowledge_base = KnowledgeBase.load(KNOWLEDGE_BASE_PATH)
tfidf_matcher = QuestionMatcher(app.state.knowledge_base)
semantic_matcher = SemanticMatcher(app.state.knowledge_base)
app.state.matcher = HybridMatcher(tfidf_matcher, semantic_matcher)

app.include_router(router)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")
