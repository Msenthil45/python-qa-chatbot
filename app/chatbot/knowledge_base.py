import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

REQUIRED_FIELDS = {"id", "category", "question", "answer"}


class KnowledgeBaseError(Exception):
    pass


class KnowledgeBase:
    def __init__(self, entries: List[Dict[str, Any]]):
        self._entries = entries

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        if not path.exists():
            raise KnowledgeBaseError(f"Knowledge base file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            try:
                raw = json.load(f)
            except json.JSONDecodeError as e:
                raise KnowledgeBaseError(f"Knowledge base file is not valid JSON: {e}") from e

        if not isinstance(raw, list):
            raise KnowledgeBaseError("Knowledge base must be a JSON array of entries")

        entries = []
        seen_ids = set()
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise KnowledgeBaseError(f"Entry at index {i} is not a JSON object")

            missing = REQUIRED_FIELDS - entry.keys()
            if missing:
                raise KnowledgeBaseError(
                    f"Entry at index {i} (id={entry.get('id', '?')}) is missing required fields: {missing}"
                )

            if entry["id"] in seen_ids:
                raise KnowledgeBaseError(f"Duplicate entry id: {entry['id']}")
            seen_ids.add(entry["id"])

            entry.setdefault("variations", [])
            entry.setdefault("code_example", "")
            entries.append(entry)

        return cls(entries)

    @property
    def entries(self) -> List[Dict[str, Any]]:
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def searchable_corpus(self) -> List[Tuple[int, str]]:
        """
        Flattens every entry's canonical question and its variations into
        (entry_index, text) pairs, so the matcher can score against every
        phrasing while still mapping a hit back to a single KB entry.
        """
        pairs: List[Tuple[int, str]] = []
        for idx, entry in enumerate(self._entries):
            pairs.append((idx, entry["question"]))
            for variation in entry["variations"]:
                pairs.append((idx, variation))
        return pairs

    def get_by_index(self, idx: int) -> Dict[str, Any]:
        return self._entries[idx]

    def get_by_id(self, entry_id: str) -> Dict[str, Any]:
        for entry in self._entries:
            if entry["id"] == entry_id:
                return entry
        raise KeyError(f"No knowledge base entry with id: {entry_id}")

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        return [e for e in self._entries if e["category"].lower() == category.lower()]

    def categories(self) -> List[str]:
        seen = []
        for entry in self._entries:
            if entry["category"] not in seen:
                seen.append(entry["category"])
        return seen
