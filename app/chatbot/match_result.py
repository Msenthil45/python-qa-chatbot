from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MatchResult:
    matched: bool
    score: float
    entry: Optional[Dict[str, Any]]
