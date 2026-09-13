import pytest

from app.chatbot.hybrid_matcher import HybridMatcher
from app.chatbot.knowledge_base import KnowledgeBase
from app.chatbot.match_result import MatchResult
from app.chatbot.matcher import QuestionMatcher
from app.chatbot.semantic_matcher import SemanticMatcher
from app.config import KNOWLEDGE_BASE_PATH


@pytest.fixture(scope="module")
def hybrid_matcher() -> HybridMatcher:
    kb = KnowledgeBase.load(KNOWLEDGE_BASE_PATH)
    tfidf_matcher = QuestionMatcher(kb)
    semantic_matcher = SemanticMatcher(kb)
    return HybridMatcher(tfidf_matcher, semantic_matcher)


class _StubMatcher:
    """A matcher stub that returns a fixed MatchResult, for isolating the
    HybridMatcher's combination logic from the real TF-IDF/embedding
    models."""

    def __init__(self, result: MatchResult):
        self._result = result

    def match(self, user_question: str) -> MatchResult:
        return self._result


def _entry(entry_id: str) -> dict:
    return {"id": entry_id, "category": "Test", "question": "Q?", "answer": "A.", "code_example": ""}


# Regression cases already proven correct via TF-IDF alone (tests/test_matcher.py)
# and via semantic search alone (tests/test_semantic_matcher.py) — the hybrid
# combination must not regress either path.
IN_SCOPE_CASES = [
    ("what is a variable in python?", "var_001"),
    ("what's the difference between a list and a tuple?", "coll_002"),
    ("how do i reverse a string in python", "str_003"),
    ("what is a lambda function", "int_004"),
    ("what does % do in python", "op_004"),
]

OUT_OF_SCOPE_CASES = [
    "how do i cook pasta",
    "what is the capital of france",
    "tell me a joke",
    "what is javascript",
]

# TF-IDF alone confidently (score 1.0) matches the WRONG entry for this
# phrasing — a perfect-confidence false positive from sparse literal word
# overlap. This is the real bug that motivated computing both matchers on
# every request instead of treating semantic search as a fallback used only
# when TF-IDF finds nothing (a fallback-only design would never revisit a
# match TF-IDF was "confident" about, wrong or not).
TFIDF_FALSE_POSITIVE_CASE = ("making a subclass reuse behavior from a parent class", "oop_003")


@pytest.mark.parametrize("question, expected_id", IN_SCOPE_CASES)
def test_hybrid_matches_expected_entry(hybrid_matcher, question, expected_id):
    result = hybrid_matcher.match(question)
    assert result.matched is True
    assert result.entry["id"] == expected_id


@pytest.mark.parametrize("question", OUT_OF_SCOPE_CASES)
def test_hybrid_out_of_scope_question_falls_back(hybrid_matcher, question):
    result = hybrid_matcher.match(question)
    assert result.matched is False
    assert result.entry is None


def test_hybrid_corrects_tfidf_false_positive(hybrid_matcher):
    question, expected_id = TFIDF_FALSE_POSITIVE_CASE
    result = hybrid_matcher.match(question)
    assert result.matched is True
    assert result.entry["id"] == expected_id


def test_hybrid_trusts_semantic_when_tfidf_finds_nothing():
    tfidf_stub = _StubMatcher(MatchResult(matched=False, score=0.0, entry=None))
    semantic_stub = _StubMatcher(MatchResult(matched=True, score=0.7, entry=_entry("sem_1")))
    hybrid = HybridMatcher(tfidf_stub, semantic_stub)

    result = hybrid.match("anything")
    assert result.matched is True
    assert result.entry["id"] == "sem_1"


def test_hybrid_trusts_tfidf_when_semantic_is_not_confident():
    tfidf_stub = _StubMatcher(MatchResult(matched=True, score=0.6, entry=_entry("tfidf_1")))
    semantic_stub = _StubMatcher(MatchResult(matched=False, score=0.2, entry=None))
    hybrid = HybridMatcher(tfidf_stub, semantic_stub)

    result = hybrid.match("anything")
    assert result.matched is True
    assert result.entry["id"] == "tfidf_1"


def test_hybrid_agrees_when_both_matchers_pick_same_entry():
    tfidf_stub = _StubMatcher(MatchResult(matched=True, score=0.9, entry=_entry("same_id")))
    semantic_stub = _StubMatcher(MatchResult(matched=True, score=0.6, entry=_entry("same_id")))
    hybrid = HybridMatcher(tfidf_stub, semantic_stub)

    result = hybrid.match("anything")
    assert result.matched is True
    assert result.entry["id"] == "same_id"


def test_hybrid_prefers_semantic_when_both_confident_but_disagree():
    tfidf_stub = _StubMatcher(MatchResult(matched=True, score=1.0, entry=_entry("tfidf_pick")))
    semantic_stub = _StubMatcher(MatchResult(matched=True, score=0.55, entry=_entry("semantic_pick")))
    hybrid = HybridMatcher(tfidf_stub, semantic_stub)

    result = hybrid.match("anything")
    assert result.matched is True
    assert result.entry["id"] == "semantic_pick"


def test_hybrid_falls_back_when_neither_matcher_is_confident():
    tfidf_stub = _StubMatcher(MatchResult(matched=False, score=0.1, entry=None))
    semantic_stub = _StubMatcher(MatchResult(matched=False, score=0.3, entry=None))
    hybrid = HybridMatcher(tfidf_stub, semantic_stub)

    result = hybrid.match("anything")
    assert result.matched is False
    assert result.entry is None
