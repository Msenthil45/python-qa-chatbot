import pytest

from app.chatbot.knowledge_base import KnowledgeBase
from app.chatbot.matcher import QuestionMatcher
from app.config import KNOWLEDGE_BASE_PATH


@pytest.fixture(scope="module")
def matcher() -> QuestionMatcher:
    kb = KnowledgeBase.load(KNOWLEDGE_BASE_PATH)
    return QuestionMatcher(kb)


IN_SCOPE_CASES = [
    ("what is a variable in python?", "var_001"),
    ("what's the difference between a list and a tuple?", "coll_002"),
    ("how do i reverse a string in python", "str_003"),
    ("what is the gil in python", "int_003"),
    ("explain inheritance in python oop", "oop_003"),
    ("how to handle exceptions try except", "exc_001"),
    ("difference between == and is", "op_002"),
    ("what is a lambda function", "int_004"),
    ("what is *args and **kwargs used for", "func_003"),
    ("how to open read a text file", "file_001"),
    ("what does % do in python", "op_004"),
    ("what is a decorator", "int_002"),
    ("deep copy vs shallow copy", "int_001"),
    ("pep8 style guide", "int_006"),
]

OUT_OF_SCOPE_CASES = [
    "how do i cook pasta",
    "what is the capital of france",
    "tell me a joke",
    "what is javascript",
    "how do i center a div in css",
]


@pytest.mark.parametrize("question, expected_id", IN_SCOPE_CASES)
def test_in_scope_question_matches_expected_entry(matcher, question, expected_id):
    result = matcher.match(question)
    assert result.matched is True
    assert result.entry["id"] == expected_id
    # allow tiny floating-point overshoot past 1.0 from cosine similarity rounding
    assert 0.0 <= result.score <= 1.0 + 1e-9


@pytest.mark.parametrize("question", OUT_OF_SCOPE_CASES)
def test_out_of_scope_question_falls_back(matcher, question):
    result = matcher.match(question)
    assert result.matched is False
    assert result.entry is None


def test_empty_question_falls_back(matcher):
    result = matcher.match("")
    assert result.matched is False
    assert result.score == 0.0
    assert result.entry is None


def test_whitespace_only_question_falls_back(matcher):
    result = matcher.match("     ")
    assert result.matched is False
    assert result.entry is None


def test_matching_is_case_insensitive(matcher):
    lower = matcher.match("what is a variable in python?")
    upper = matcher.match("WHAT IS A VARIABLE IN PYTHON?")
    assert lower.matched is True
    assert upper.matched is True
    assert lower.entry["id"] == upper.entry["id"]


def test_matched_entry_has_expected_shape(matcher):
    result = matcher.match("what is a variable in python?")
    assert result.matched is True
    entry = result.entry
    for field in ("id", "category", "question", "answer", "code_example"):
        assert field in entry


def test_matcher_raises_on_empty_knowledge_base():
    empty_kb = KnowledgeBase([])
    with pytest.raises(ValueError):
        QuestionMatcher(empty_kb)


def test_custom_threshold_can_make_matching_stricter():
    kb = KnowledgeBase.load(KNOWLEDGE_BASE_PATH)
    strict_matcher = QuestionMatcher(kb, threshold=0.99)
    result = strict_matcher.match("explain inheritance in python oop")
    # score for this phrasing (~0.89) is below a 0.99 threshold
    assert result.matched is False
