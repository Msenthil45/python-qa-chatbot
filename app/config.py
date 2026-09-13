from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_PATH = BASE_DIR / "data" / "knowledge_base.json"
FRONTEND_DIR = BASE_DIR / "frontend"

# Minimum cosine similarity score (0.0 - 1.0) required before we trust a match.
# Anything below this returns the fallback response instead of a guess.
SIMILARITY_THRESHOLD = 0.35
