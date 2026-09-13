from fastapi.testclient import TestClient

from app.api.routes import FALLBACK_MESSAGE
from app.main import app

client = TestClient(app)


def test_chat_with_in_scope_question_returns_matched_answer():
    response = client.post("/api/chat", json={"message": "what is a variable in python?"})
    assert response.status_code == 200

    data = response.json()
    assert data["matched"] is True
    assert data["category"] == "Variables"
    assert "variable" in data["answer"].lower()
    assert data["code_example"]
    assert data["matched_question"]
    assert 0.0 <= data["confidence"] <= 1.0 + 1e-9


def test_chat_with_out_of_scope_question_returns_fallback():
    response = client.post("/api/chat", json={"message": "how do i cook pasta"})
    assert response.status_code == 200

    data = response.json()
    assert data["matched"] is False
    assert data["answer"] == FALLBACK_MESSAGE
    assert data["category"] is None
    assert data["code_example"] is None
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


def test_chat_response_answer_is_never_empty():
    for message in ["what is a decorator", "how do i cook pasta"]:
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
