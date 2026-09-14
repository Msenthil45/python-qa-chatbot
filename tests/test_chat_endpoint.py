import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.routes import FALLBACK_MESSAGE
from app.chatbot.llm_client import LLMAnswer
from app.config import OLLAMA_BASE_URL
from app.main import app

client = TestClient(app)

# A question already verified (tests/test_matcher.py, tests/test_hybrid_matcher.py)
# to score 0 against the Python knowledge base, so it always reaches the LLM path
# below regardless of matcher tuning.
OUT_OF_SCOPE_QUESTION = "how do i cook pasta"


class _StubLLMClient:
    def __init__(self, result: LLMAnswer):
        self._result = result

    def generate_answer(self, question: str) -> LLMAnswer:
        return self._result


@pytest.fixture
def stub_llm(monkeypatch):
    """Replaces app.state.llm_client so tests don't depend on a running
    Ollama server and don't wait on real model generation time."""

    def _set(result: LLMAnswer):
        monkeypatch.setattr(app.state, "llm_client", _StubLLMClient(result))

    return _set


def test_chat_with_in_scope_question_returns_matched_answer():
    response = client.post("/api/chat", json={"message": "what is a variable in python?"})
    assert response.status_code == 200

    data = response.json()
    assert data["matched"] is True
    assert data["source"] == "knowledge_base"
    assert data["category"] == "Variables"
    assert "variable" in data["answer"].lower()
    assert data["code_example"]
    assert data["matched_question"]
    assert 0.0 <= data["confidence"] <= 1.0 + 1e-9


def test_chat_out_of_scope_question_falls_back_when_llm_unavailable(stub_llm):
    stub_llm(LLMAnswer(answer="", code_example=None, error="connection refused"))

    response = client.post("/api/chat", json={"message": OUT_OF_SCOPE_QUESTION})
    assert response.status_code == 200

    data = response.json()
    assert data["matched"] is False
    assert data["source"] == "none"
    assert data["answer"] == FALLBACK_MESSAGE
    assert data["category"] is None
    assert data["code_example"] is None
    assert data["matched_question"] is None


def test_chat_out_of_scope_question_uses_llm_when_available(stub_llm):
    stub_llm(
        LLMAnswer(
            answer="A closure is a function that remembers variables from its outer scope.",
            code_example="function f() {\n  return 1;\n}",
            error=None,
        )
    )

    response = client.post("/api/chat", json={"message": "what is a closure in javascript"})
    assert response.status_code == 200

    data = response.json()
    assert data["matched"] is False
    assert data["source"] == "ai"
    assert "closure" in data["answer"].lower()
    assert data["code_example"] == "function f() {\n  return 1;\n}"
    assert data["category"] is None
    assert data["matched_question"] is None


def test_chat_with_empty_message_returns_422():
    response = client.post("/api/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_with_whitespace_only_message_returns_422():
    response = client.post("/api/chat", json={"message": "     "})
    assert response.status_code == 422


def test_chat_with_missing_message_field_returns_422():
    response = client.post("/api/chat", json={})
    assert response.status_code == 422


def test_chat_with_overly_long_message_returns_422():
    response = client.post("/api/chat", json={"message": "a" * 501})
    assert response.status_code == 422


def test_chat_with_non_string_message_returns_422():
    response = client.post("/api/chat", json={"message": 12345})
    assert response.status_code == 422


def test_chat_response_answer_is_never_empty(stub_llm):
    stub_llm(LLMAnswer(answer="Here's a short answer.", code_example=None, error=None))

    for message in ["what is a decorator", OUT_OF_SCOPE_QUESTION]:
        response = client.post("/api/chat", json={"message": message})
        assert response.status_code == 200
        assert response.json()["answer"].strip() != ""


def test_categories_endpoint_lists_all_expected_categories():
    response = client.get("/api/categories")
    assert response.status_code == 200

    categories = set(response.json()["categories"])
    expected = {
        "Variables",
        "Data Types",
        "Operators",
        "Conditional Statements",
        "Loops",
        "Functions",
        "Lists, Tuples, Sets, and Dictionaries",
        "Strings",
        "Object-Oriented Programming",
        "Exception Handling",
        "File Handling",
        "Modules and Packages",
        "Common Python Interview Questions",
    }
    assert expected.issubset(categories)


def _ollama_is_reachable() -> bool:
    try:
        httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return True
    except httpx.HTTPError:
        return False


@pytest.mark.skipif(not _ollama_is_reachable(), reason="Ollama is not running locally")
def test_chat_live_llm_answers_non_python_question():
    """Real end-to-end smoke test against the actual local Ollama server —
    skipped automatically if Ollama isn't reachable, so the rest of the
    suite never depends on it being installed/running."""
    response = client.post("/api/chat", json={"message": "What is a pointer in C?"})
    assert response.status_code == 200

    data = response.json()
    assert data["source"] == "ai"
    assert data["answer"].strip() != ""
