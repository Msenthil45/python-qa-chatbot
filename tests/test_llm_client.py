import httpx
import pytest

from app.chatbot.llm_client import LLMClient, _split_answer_and_code


def test_split_answer_and_code_extracts_fenced_block():
    raw = "A closure is a function that remembers its outer scope.\n\n```javascript\nfunction foo() {}\n```"
    answer, code = _split_answer_and_code(raw)
    assert "closure" in answer.lower()
    assert code == "function foo() {}"


def test_split_answer_and_code_without_code_block():
    raw = "Just a plain explanation with no code."
    answer, code = _split_answer_and_code(raw)
    assert answer == raw
    assert code is None


def test_split_answer_and_code_strips_language_tag():
    raw = "Explanation.\n```python\nprint('hi')\n```"
    _, code = _split_answer_and_code(raw)
    assert code == "print('hi')"


def test_split_answer_and_code_handles_block_with_no_language_tag():
    raw = "Explanation.\n```\nplain code\n```"
    _, code = _split_answer_and_code(raw)
    assert code == "plain code"


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._json_data


def test_generate_answer_success(monkeypatch):
    def fake_post(url, json, timeout):
        assert "closure" in json["prompt"].lower()
        return _FakeResponse(
            {"response": "A closure remembers its outer scope.\n```js\nconst x = 1;\n```"}
        )

    monkeypatch.setattr("app.chatbot.llm_client.httpx.post", fake_post)

    client = LLMClient()
    result = client.generate_answer("what is a closure in javascript")

    assert result.error is None
    assert "closure" in result.answer.lower()
    assert result.code_example == "const x = 1;"


def test_generate_answer_handles_connection_error(monkeypatch):
    def fake_post(url, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("app.chatbot.llm_client.httpx.post", fake_post)

    client = LLMClient()
    result = client.generate_answer("anything")

    assert result.error is not None
    assert result.answer == ""
    assert result.code_example is None


def test_generate_answer_handles_http_status_error(monkeypatch):
    def fake_post(url, json, timeout):
        return _FakeResponse({}, status_code=500)

    monkeypatch.setattr("app.chatbot.llm_client.httpx.post", fake_post)

    client = LLMClient()
    result = client.generate_answer("anything")

    assert result.error is not None


def test_generate_answer_handles_empty_response(monkeypatch):
    def fake_post(url, json, timeout):
        return _FakeResponse({"response": "   "})

    monkeypatch.setattr("app.chatbot.llm_client.httpx.post", fake_post)

    client = LLMClient()
    result = client.generate_answer("anything")

    assert result.error is not None
    assert result.answer == ""


def test_generate_answer_sends_expected_request_shape(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse({"response": "ok"})

    monkeypatch.setattr("app.chatbot.llm_client.httpx.post", fake_post)

    client = LLMClient(base_url="http://localhost:11434", model="qwen3:8b", timeout_seconds=30)
    client.generate_answer("what is a pointer in c")

    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["json"]["model"] == "qwen3:8b"
    assert captured["json"]["stream"] is False
    assert captured["json"]["think"] is False
    assert captured["timeout"] == 30
