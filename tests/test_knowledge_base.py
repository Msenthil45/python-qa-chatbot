import json

import pytest

from app.chatbot.knowledge_base import KnowledgeBase, KnowledgeBaseError
from app.config import KNOWLEDGE_BASE_PATH


def test_loads_real_knowledge_base_file():
    kb = KnowledgeBase.load(KNOWLEDGE_BASE_PATH)
    assert len(kb) > 0


def test_all_required_categories_present():
    kb = KnowledgeBase.load(KNOWLEDGE_BASE_PATH)
    expected_categories = {
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
    assert expected_categories.issubset(set(kb.categories()))


def test_every_entry_has_a_code_example():
    kb = KnowledgeBase.load(KNOWLEDGE_BASE_PATH)
    empty = [e["id"] for e in kb.entries if not e["code_example"].strip()]
    assert empty == []


def test_get_by_id_returns_matching_entry():
    kb = KnowledgeBase.load(KNOWLEDGE_BASE_PATH)
    entry = kb.get_by_id("var_001")
    assert entry["category"] == "Variables"
    assert "variable" in entry["question"].lower()


def test_get_by_id_missing_raises_key_error():
    kb = KnowledgeBase.load(KNOWLEDGE_BASE_PATH)
    with pytest.raises(KeyError):
        kb.get_by_id("does_not_exist")


def test_get_by_category_filters_correctly():
    kb = KnowledgeBase.load(KNOWLEDGE_BASE_PATH)
    variable_entries = kb.get_by_category("Variables")
    assert len(variable_entries) > 0
    assert all(e["category"] == "Variables" for e in variable_entries)


def test_searchable_corpus_maps_back_to_correct_entry():
    kb = KnowledgeBase.load(KNOWLEDGE_BASE_PATH)
    corpus = kb.searchable_corpus()

    assert len(corpus) >= len(kb)

    for idx, text in corpus:
        entry = kb.get_by_index(idx)
        assert text == entry["question"] or text in entry["variations"]


def test_missing_file_raises_knowledge_base_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    with pytest.raises(KnowledgeBaseError):
        KnowledgeBase.load(missing_path)


def test_invalid_json_raises_knowledge_base_error(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(KnowledgeBaseError):
        KnowledgeBase.load(bad_file)


def test_non_list_json_raises_knowledge_base_error(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(KnowledgeBaseError):
        KnowledgeBase.load(bad_file)


def test_entry_missing_required_field_raises_error(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(
        json.dumps([{"id": "x1", "category": "Test", "question": "Q?"}]),  # no "answer"
        encoding="utf-8",
    )
    with pytest.raises(KnowledgeBaseError):
        KnowledgeBase.load(bad_file)


def test_duplicate_id_raises_error(tmp_path):
    bad_file = tmp_path / "bad.json"
    entry = {"id": "dup_1", "category": "Test", "question": "Q?", "answer": "A."}
    bad_file.write_text(json.dumps([entry, entry]), encoding="utf-8")
    with pytest.raises(KnowledgeBaseError):
        KnowledgeBase.load(bad_file)


def test_defaults_are_applied_for_optional_fields(tmp_path):
    minimal_file = tmp_path / "minimal.json"
    minimal_file.write_text(
        json.dumps([{"id": "x1", "category": "Test", "question": "Q?", "answer": "A."}]),
        encoding="utf-8",
    )
    kb = KnowledgeBase.load(minimal_file)
    entry = kb.get_by_index(0)
    assert entry["variations"] == []
    assert entry["code_example"] == ""
