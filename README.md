# Python Q&A Chatbot

A knowledge-base-driven chatbot that answers Python programming questions using a hybrid of lexical (TF-IDF + cosine similarity) and semantic (sentence-embedding) natural-language matching over a curated, hand-written Q&A dataset. Built with FastAPI and a vanilla HTML/CSS/JS frontend — **no paid APIs, no external LLM calls**. If a question isn't in its knowledge base, it says so honestly instead of guessing.

## Why this project

Most "AI chatbot" tutorials wrap an LLM API call and call it done. This one deliberately doesn't: it's a small, understandable NLP pipeline that can only ever answer from a knowledge base it was given, which makes its behavior fully predictable and testable. That constraint is the point — it demonstrates knowledge-base design, TF-IDF/cosine-similarity matching, a clean FastAPI service layer, and a responsive frontend, without hiding the logic behind someone else's model.

## Live demo

Run it locally (see [Setup](#setup) below) and open `http://127.0.0.1:8000`.

## Architecture

```
Browser (chat UI)
   │  POST /api/chat  { "message": "what is a list in python?" }
   ▼
FastAPI app (app/main.py)
   │
   ├─────────────────────────────┬─────────────────────────────────┐
   ▼                              ▼
TF-IDF matcher                 Semantic matcher
(app/chatbot/matcher.py)       (app/chatbot/semantic_matcher.py)
lowercase, spell out bare      sentence-transformers
operators (==, !=, //, %),     (all-MiniLM-L6-v2) embeddings +
then TF-IDF + cosine           cosine similarity — catches
similarity against every       paraphrases that share no
question + phrasing variation  literal words with the KB
   │                              │
   └─────────────┬────────────────┘
                 ▼
      HybridMatcher combination rule (app/chatbot/hybrid_matcher.py):
        - TF-IDF found nothing            → trust semantic
        - semantic isn't confident enough → trust TF-IDF
        - both agree on the same entry    → return it
        - both matched, disagree          → trust semantic
                 │
                 ▼
     matched?  ──yes──▶  return the KB answer + code example
        │
        no
        ▼
   return an honest fallback message — never a fabricated answer
   │
   ▼
JSON response  →  rendered in the chat UI, appended to session history
```

Key design decisions:

- **Knowledge base as JSON, not a database.** Easy to read, diff, and hand-edit; a database only earns its place once the KB needs concurrent multi-user editing (see [Future Improvements](#future-improvements)).
- **TF-IDF + cosine similarity as the primary matcher, not a live LLM.** Deterministic, free, fast, and honest — it can only return what's in the knowledge base, never invent a Python fact.
- **Semantic search runs on every request, not just as a fallback.** The obvious design is "try TF-IDF, only run the embedding model if TF-IDF finds nothing" — but TF-IDF can return a *wrong* entry with deceptively high (even 1.0) confidence purely from sparse literal word overlap, and a fallback-only design would never get a chance to catch that. Running both and letting semantic search override a confident-but-disagreeing TF-IDF pick fixed a real bug found during development (see [Known Issues Found & Fixed](#known-issues-found--fixed-during-development)).
- **Clean separation of concerns.** `app/chatbot/` (pure Python — no web framework) is unit-testable in isolation from `app/api/` (the FastAPI glue) and `app/main.py` (wiring).

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Python 3.12+ (tested on 3.13), FastAPI, Uvicorn |
| Lexical matching | scikit-learn (`TfidfVectorizer` + `cosine_similarity`) |
| Semantic matching | `sentence-transformers` (`all-MiniLM-L6-v2`) + cosine similarity |
| Data | Static JSON knowledge base |
| Frontend | Vanilla HTML / CSS / JavaScript — no framework, no build step |
| Testing | pytest, FastAPI `TestClient` |

## Project structure

```
python-qa-chatbot/
├── app/
│   ├── main.py                  # FastAPI app: builds KB + matchers, mounts static files, serves index.html
│   ├── config.py                # Paths + similarity thresholds + embedding model name
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   ├── chatbot/
│   │   ├── preprocessor.py      # Text normalization (lowercase, operator spelling, whitespace)
│   │   ├── knowledge_base.py    # Loads + validates data/knowledge_base.json
│   │   ├── match_result.py      # Shared MatchResult dataclass
│   │   ├── matcher.py           # TF-IDF + cosine similarity question matching
│   │   ├── semantic_matcher.py  # Sentence-embedding + cosine similarity matching
│   │   └── hybrid_matcher.py    # Combines both matchers (see Architecture above)
│   └── api/
│       └── routes.py            # /api/chat, /api/categories, /api/health
├── data/
│   └── knowledge_base.json      # 54 Q&A entries across 13 Python topics
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── tests/
│   ├── test_knowledge_base.py
│   ├── test_matcher.py
│   ├── test_semantic_matcher.py
│   ├── test_hybrid_matcher.py
│   ├── test_chat_endpoint.py
│   └── test_api.py
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.12 or newer.

```bash
python -m venv venv
```

Activate it (PowerShell):

```bash
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

This includes `sentence-transformers` (and its dependency `torch`), so the install is a few hundred MB and can take a few minutes. On the very first run, the app also downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) from Hugging Face — this needs an internet connection once; after that it's cached locally (`~/.cache/huggingface`) and startup is fast.

Run the server:

```bash
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000** in your browser. Interactive API docs are at **http://127.0.0.1:8000/docs**.

### Running in VS Code

1. Open the project folder in VS Code.
2. Select the `venv` interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → choose `.\venv\Scripts\python.exe`.
3. Open a terminal (`` Ctrl+` ``) — it should auto-activate the venv — and run the `uvicorn` command above.
4. `Ctrl+C` in the terminal stops the server.

## Running the tests

```bash
pytest -v
```

73 tests across 6 files, covering: knowledge base loading/validation; TF-IDF matching accuracy and fallback behavior; semantic (embedding) matching accuracy, including paraphrases with zero literal word overlap; the hybrid combination logic (both with the real models and with stubbed matchers to isolate the decision rule); and full HTTP request/response validation for the chat endpoint.

The semantic and hybrid tests load the real `all-MiniLM-L6-v2` model, so the first `pytest` run will be slower if the model isn't cached yet (see [Setup](#setup)).

## API reference

### `POST /api/chat`

Request:
```json
{ "message": "What is a decorator in Python?" }
```

Response (200):
```json
{
  "answer": "A decorator is a function that wraps another function...",
  "matched": true,
  "confidence": 1.0,
  "category": "Common Python Interview Questions",
  "code_example": "def shout(func):\n    ...",
  "matched_question": "What are decorators in Python?"
}
```

`message` must be 1–500 non-blank characters, or the API returns `422 Unprocessable Entity`.

### `GET /api/categories`

Returns the list of knowledge base categories.

### `GET /api/health`

Returns `{"status": "ok"}` — useful for uptime checks.

## Sample questions and expected behavior

| You ask | Category matched | Confidence |
|---|---|---|
| "What is a variable in Python?" | Variables | ~1.00 |
| "What's the difference between a list and a tuple?" | Lists, Tuples, Sets, and Dictionaries | ~1.00 |
| "How do I reverse a string in Python?" | Strings | ~1.00 |
| "What is the GIL in Python?" | Common Python Interview Questions | ~1.00 |
| "Explain inheritance in Python OOP" | Object-Oriented Programming | ~0.89 |
| "What does % do in Python?" | Operators | ~0.77 |
| "Put the characters of a word in the opposite order" *(paraphrase, zero literal overlap with the KB — caught only by semantic search)* | Strings | ~0.51 |
| "How do I cook pasta?" | *(none — fallback message)* | 0.00 |
| "What is the capital of France?" | *(none — fallback message)* | 0.00 |

Confidence is cosine similarity in `[0, 1]`. TF-IDF matches below `SIMILARITY_THRESHOLD` (0.35) and semantic matches below `SEMANTIC_SIMILARITY_THRESHOLD` (0.5) don't count as confident (both in `app/config.py`); if neither matcher is confident, the response is the fallback instead of a low-confidence guess.

## Known issues found & fixed during development

Real matching bugs surfaced while building the NLP layer — worth documenting because they're the actual reason the matcher looks the way it does:

1. **Generic words inflated false matches.** Without stop-word removal, a nonsense query like "how do I cook pasta" scored 0.54 against an unrelated KB entry — cosine similarity on two "all common words" vectors can look deceptively similar in direction even when unrelated in meaning. Fixed by filtering English stop words (plus the domain-specific word "python", which appears in nearly every entry and carries no discriminating signal).
2. **Symbol-only operators vanished during tokenization.** `==`, `!=`, `<=`, `>=`, `//`, and `%` have no letters or digits, so scikit-learn's default tokenizer dropped them entirely — a question like "difference between == and is" lost all its distinguishing signal and matched the wrong entry with false 100% confidence. Fixed in `app/chatbot/preprocessor.py` by spelling out operators into words (`==` → "equals") before vectorization, applied identically to the knowledge base and every user query.
3. **TF-IDF's own confident matches could be wrong.** After adding semantic search, the first design ran it only as a fallback when TF-IDF found *nothing* — but "making a subclass reuse behavior from a parent class" matched a "what is a class?" entry via TF-IDF at a perfect 1.0 score, purely from sparse literal overlap having nothing else to compete against. A fallback-only design never revisits a match TF-IDF was "confident" about, wrong or not. Fixed by having `HybridMatcher` always compute both matchers and let semantic search override when it disagrees with TF-IDF and clears its own confidence threshold (see `app/chatbot/hybrid_matcher.py`).

## Limitations

- Answers only what's in the knowledge base — 54 entries across 13 topics. It has no general Python knowledge beyond that.
- Even with semantic search, matching is still limited by a small general-purpose embedding model and only 54 KB entries — some ambiguous paraphrases still resolve to a plausible-but-not-ideal entry (e.g. a "list comprehension" question landing on a general "list methods" entry) rather than the perfect match. This is an inherent limitation of the approach at this scale, not a bug to chase indefinitely.
- Every chat request now runs the embedding model in addition to TF-IDF (since `HybridMatcher` always computes both — see [Architecture](#architecture)), adding roughly 50–150ms of CPU-bound latency per request compared to TF-IDF alone. Still well within interactive chat speed, but worth knowing if scaling to heavy concurrent load.
- Chat history is per-browser-tab (`sessionStorage`) — it resets when the tab closes and isn't shared across devices or persisted server-side.
- No authentication or rate limiting — not intended for public unauthenticated deployment as-is.
- No knowledge-base editing UI yet — updates mean hand-editing `data/knowledge_base.json`.

## Future improvements

- **Admin UI for knowledge-base management** — authenticated CRUD endpoints plus a small admin page to add/edit/remove Q&A entries without hand-editing JSON.
- **Persistent chat history** — an optional SQLite-backed history endpoint for users who want their conversations to survive across sessions/devices.
- **Expanded knowledge base** — more entries per topic, which would also sharpen semantic matching (more, denser examples per concept reduce ambiguous near-ties between entries).
- **Multi-turn follow-up questions** — e.g. "what about tuples?" after a list question, which needs conversation context the current stateless matcher doesn't keep.
- **A larger or fine-tuned embedding model** — `all-MiniLM-L6-v2` is small and fast on CPU but general-purpose; a model fine-tuned on Q&A-style pairs (or simply a bigger one) would likely resolve more of the ambiguous paraphrase cases noted above.
- **Deployment** — containerize with Docker and add a production ASGI setup (e.g. Uvicorn + Gunicorn workers) behind HTTPS.

## Screenshots

Not committed to the repo (to keep it lightweight), but easy to reproduce in under a minute:

1. `uvicorn app.main:app --reload`
2. Open `http://127.0.0.1:8000`
3. Click a suggested question, or ask your own — the resulting bubbles (with category tag, answer, and a copyable code block) make a good desktop screenshot.
4. In Chrome/Edge DevTools, toggle device toolbar (`Ctrl+Shift+M`) and pick a phone preset for a mobile screenshot.

## License

Personal portfolio project — no license file included. Feel free to ask before reusing substantial portions.
