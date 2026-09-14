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

# When neither matcher finds a confident answer in the curated Python
# knowledge base, the app falls back to a local LLM (via Ollama) so it can
# still answer questions about other programming languages and concepts
# outside the fixed KB. This never calls a paid/external API — Ollama runs
# entirely on the local machine.
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_TIMEOUT_SECONDS = 45
