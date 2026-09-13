from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_PATH = BASE_DIR / "data" / "knowledge_base.json"
FRONTEND_DIR = BASE_DIR / "frontend"

# Minimum cosine similarity score (0.0 - 1.0) required before we trust a match.
# Anything below this returns the fallback response instead of a guess.
SIMILARITY_THRESHOLD = 0.35

# Small, fast sentence-embedding model used as a semantic second pass when
# TF-IDF finds no confident lexical match — it can catch paraphrases that
# share few literal words with the knowledge base.
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Embedding cosine similarities run on a different scale than TF-IDF's, so
# this uses its own, higher-bar threshold to avoid false positives on
# genuinely out-of-scope questions.
SEMANTIC_SIMILARITY_THRESHOLD = 0.5
