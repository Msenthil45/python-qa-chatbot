import pytest

from app.chatbot.knowledge_base import KnowledgeBase
from app.chatbot.semantic_matcher import SemanticMatcher
from app.config import KNOWLEDGE_BASE_PATH


@pytest.fixture(scope="module")
def semantic_matcher() -> SemanticMatcher:
    kb = KnowledgeBase.load(KNOWLEDGE_BASE_PATH)
    return SemanticMatcher(kb)


# A mix of direct questions (semantic search should still get these right)
# and genuine paraphrases with ~zero literal word overlap with the matching
# entry — verified separately that TF-IDF alone scores 0.000 on these — but
# clear conceptual overlap a human would recognize immediately.
DIRECT_CASES = [
    ("what is a variable in python?", "var_001"),
    ("how do i reverse a string in python", "str_003"),
]

PARAPHRASE_CASES = [
    ("put the characters of a word in the opposite order", "str_003"),
]

OUT_OF_SCOPE_CASES = [
    "how do i cook pasta",
    "what is the capital of france",
    "tell me a joke",
]


@pytest.mark.parametrize("question, expected_id", DIRECT_CASES)
def test_semantic_match_finds_correct_entry_for_direct_questions(semantic_matcher, question, expected_id):
    result = semantic_matcher.match(question)
    assert result.matched is True
    assert result.entry["id"] == expected_id
    assert 0.0 <= result.score <= 1.0 + 1e-9


@pytest.mark.parametrize("question, expected_id", PARAPHRASE_CASES)
def test_semantic_match_finds_correct_entry_for_paraphrases(semantic_matcher, question, expected_id):
    result = semantic_matcher.match(question)
    assert result.matched is True
    assert result.entry["id"] == expected_id


@pytest.mark.parametrize("question", OUT_OF_SCOPE_CASES)
def test_semantic_out_of_scope_question_falls_back(semantic_matcher, question):
    result = semantic_matcher.match(question)
    assert result.matched is False
    assert result.entry is None


def test_empty_question_falls_back(semantic_matcher):
    result = semantic_matcher.match("")
    assert result.matched is False
    assert result.score == 0.0
    assert result.entry is None


def test_whitespace_only_question_falls_back(semantic_matcher):
    result = semantic_matcher.match("     ")
    assert result.matched is False
    assert result.entry is None


def test_matcher_raises_on_empty_knowledge_base():
    empty_kb = KnowledgeBase([])
    with pytest.raises(ValueError):
        SemanticMatcher(empty_kb)


def test_custom_threshold_can_make_matching_stricter():
    kb = KnowledgeBase.load(KNOWLEDGE_BASE_PATH)
    strict_matcher = SemanticMatcher(kb, threshold=0.999)
    result = strict_matcher.match("how do i reverse a string in python")
    assert result.matched is False
