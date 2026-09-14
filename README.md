# Python Q&A Chatbot

A chatbot with two tiers: verified Python answers from a curated knowledge base (matched via a hybrid of TF-IDF and sentence-embedding similarity), and a local LLM fallback (via [Ollama](https://ollama.com), no cloud API, no API key) for every other language and concept. Built with FastAPI and a vanilla HTML/CSS/JS frontend. Every response is tagged with where it came from, so the UI never presents a generated guess as a verified fact.

## Why this project

Most "AI chatbot" tutorials wrap an LLM API call and call it done, which makes correctness impossible to reason about — you can't unit test what a hosted model will say. This project started as the opposite: a small, fully deterministic NLP pipeline (TF-IDF + cosine similarity, then sentence embeddings) that could only ever answer from a knowledge base it was explicitly given, and said so honestly when a question fell outside it.

That's still the core of the app for Python questions — and it's why the matching logic is unit-tested in isolation with real accuracy assertions, not just "does it return 200." But a knowledge base of 54 hand-written entries can't cover "every programming language and concept," so the fallback path now hands off to a small model running locally through Ollama — free, no data leaving the machine, and clearly labeled in the UI as AI-generated rather than verified. The interesting engineering problem became: how do you combine a system that's provably correct-or-honest with one that can be wrong, without letting the second one quietly borrow the first one's credibility?

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
     matched?  ──yes──▶  return the verified KB answer + code example (source: "knowledge_base")
        │
        no
        ▼
   LLMClient.generate_answer() — POST http://localhost:11434/api/generate (Ollama, local)
        │
   Ollama reachable and responded?
        │
   ┌────┴────┐
  yes         no
   │           │
   ▼           ▼
return the    return the honest static fallback message
AI-generated  (never a fabricated "verified" answer)
answer        (source: "none")
(source: "ai")
        │
        ▼
JSON response  →  rendered in the chat UI, appended to session history
                   (AI-generated answers get a distinct "AI generated" tag,
                    never the KB's category tag)
```

Key design decisions:

- **Knowledge base as JSON, not a database.** Easy to read, diff, and hand-edit; a database only earns its place once the KB needs concurrent multi-user editing (see [Future Improvements](#future-improvements)).
- **TF-IDF + cosine similarity as the primary matcher, not an LLM.** Deterministic, fast, and honest for the topics it covers — it can only return what's in the knowledge base, never invent a Python fact.
- **Semantic search runs on every request, not just as a fallback.** The obvious design is "try TF-IDF, only run the embedding model if TF-IDF finds nothing" — but TF-IDF can return a *wrong* entry with deceptively high (even 1.0) confidence purely from sparse literal word overlap, and a fallback-only design would never get a chance to catch that. Running both and letting semantic search override a confident-but-disagreeing TF-IDF pick fixed a real bug found during development (see [Known Issues Found & Fixed](#known-issues-found--fixed-during-development)).
- **The LLM fallback is local-only, and every response says where it came from.** `POST /api/chat` always returns a `source` field (`"knowledge_base"`, `"ai"`, or `"none"`) alongside `matched`. The frontend uses it to render a distinct "AI generated" tag instead of a category tag — an AI answer is never presented with the same visual trust signal as a knowledge-base one. No cloud API, no API key, nothing sent off the machine.
- **Clean separation of concerns.** `app/chatbot/` (pure Python — no web framework) is unit-testable in isolation from `app/api/` (the FastAPI glue) and `app/main.py` (wiring).

## Tech stack

| Layer | Choice |
|---|---|
| Backend | Python 3.12+ (tested on 3.13), FastAPI, Uvicorn |
| Lexical matching | scikit-learn (`TfidfVectorizer` + `cosine_similarity`) |
| Semantic matching | `sentence-transformers` (`all-MiniLM-L6-v2`) + cosine similarity |
| LLM fallback | [Ollama](https://ollama.com) running locally (`qwen3:8b` by default) via its REST API |
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
│   │   ├── hybrid_matcher.py    # Combines both matchers (see Architecture above)
│   │   └── llm_client.py        # Local Ollama client — the fallback for anything outside the KB
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
│   ├── test_llm_client.py
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

### Setting up Ollama (for the AI fallback)

This is optional — the app runs fine without it, and the Python knowledge base works exactly the same either way. Without Ollama running, questions outside the KB just get the honest static fallback message instead of an AI-generated answer.

1. Install [Ollama](https://ollama.com/download) (or `winget install Ollama.Ollama` on Windows).
2. Pull the default model:
   ```bash
   ollama pull qwen3:8b
   ```
   This is a ~5.2 GB download. To use a smaller/faster model instead, pull it (e.g. `ollama pull qwen2.5-coder:3b`) and change `OLLAMA_MODEL` in `app/config.py`.
3. Ollama runs as a background service after install — `ollama list` should show the model. The app talks to it at `http://localhost:11434` (also configurable in `app/config.py`).

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

84 tests across 7 files, covering: knowledge base loading/validation; TF-IDF matching accuracy and fallback behavior; semantic (embedding) matching accuracy, including paraphrases with zero literal word overlap; the hybrid combination logic (both with the real models and with stubbed matchers to isolate the decision rule); the LLM client's response parsing and error handling (fully mocked — no network); and full HTTP request/response validation for the chat endpoint (with the LLM client stubbed, so these are fast and deterministic).

The semantic and hybrid tests load the real `all-MiniLM-L6-v2` model, so the first `pytest` run will be slower if the model isn't cached yet (see [Setup](#setup)). One additional test, `test_chat_live_llm_answers_non_python_question`, makes a real call to a local Ollama server end-to-end — it auto-skips if Ollama isn't reachable, so the rest of the suite never depends on it being installed or running.

## API reference

### `POST /api/chat`

Request:
```json
{ "message": "What is a decorator in Python?" }
```

Response (200) — a verified knowledge-base match:
```json
{
  "answer": "A decorator is a function that wraps another function...",
  "matched": true,
  "confidence": 1.0,
  "category": "Common Python Interview Questions",
  "code_example": "def shout(func):\n    ...",
  "matched_question": "What are decorators in Python?",
  "source": "knowledge_base"
}
```

Response (200) — outside the KB, answered by the local LLM instead:
```json
{
  "answer": "A closure is a function that retains access to variables from its enclosing scope...",
  "matched": false,
  "confidence": 0.31,
  "category": null,
  "code_example": "function outer() {\n  let x = 1;\n  return () => x;\n}",
  "matched_question": null,
  "source": "ai"
}
```

`source` is always one of `"knowledge_base"` (verified), `"ai"` (generated locally, may be wrong), or `"none"` (neither could answer — Ollama unreachable, so the static fallback message is returned). `message` must be 1–500 non-blank characters, or the API returns `422 Unprocessable Entity`.

### `GET /api/categories`

Returns the list of knowledge base categories.

### `GET /api/health`

Returns `{"status": "ok"}` — useful for uptime checks.

## Sample questions and expected behavior

| You ask | Source | Category / confidence |
|---|---|---|
| "What is a variable in Python?" | `knowledge_base` | Variables, ~1.00 |
| "What's the difference between a list and a tuple?" | `knowledge_base` | Lists, Tuples, Sets, and Dictionaries, ~1.00 |
| "How do I reverse a string in Python?" | `knowledge_base` | Strings, ~1.00 |
| "What is the GIL in Python?" | `knowledge_base` | Common Python Interview Questions, ~1.00 |
| "Explain inheritance in Python OOP" | `knowledge_base` | Object-Oriented Programming, ~0.89 |
| "Put the characters of a word in the opposite order" *(paraphrase, zero literal overlap with the KB — caught only by semantic search)* | `knowledge_base` | Strings, ~0.51 |
| "What is a closure in JavaScript?" | `ai` | *(outside the Python KB — answered by the local LLM)* |
| "What is a pointer in C?" | `ai` | *(outside the Python KB — answered by the local LLM)* |
| "How do I cook pasta?" | `ai` or `none` | *(not a programming question — the LLM declines, or the static fallback if Ollama isn't running)* |

Confidence is cosine similarity in `[0, 1]` and only applies to `knowledge_base` matches. TF-IDF matches below `SIMILARITY_THRESHOLD` (0.35) and semantic matches below `SEMANTIC_SIMILARITY_THRESHOLD` (0.5) don't count as confident (both in `app/config.py`); if neither matcher is confident, the question is routed to the local LLM instead of returning a low-confidence guess.

## Known issues found & fixed during development

Real matching bugs surfaced while building the NLP layer — worth documenting because they're the actual reason the matcher looks the way it does:

1. **Generic words inflated false matches.** Without stop-word removal, a nonsense query like "how do I cook pasta" scored 0.54 against an unrelated KB entry — cosine similarity on two "all common words" vectors can look deceptively similar in direction even when unrelated in meaning. Fixed by filtering English stop words (plus the domain-specific word "python", which appears in nearly every entry and carries no discriminating signal).
2. **Symbol-only operators vanished during tokenization.** `==`, `!=`, `<=`, `>=`, `//`, and `%` have no letters or digits, so scikit-learn's default tokenizer dropped them entirely — a question like "difference between == and is" lost all its distinguishing signal and matched the wrong entry with false 100% confidence. Fixed in `app/chatbot/preprocessor.py` by spelling out operators into words (`==` → "equals") before vectorization, applied identically to the knowledge base and every user query.
3. **TF-IDF's own confident matches could be wrong.** After adding semantic search, the first design ran it only as a fallback when TF-IDF found *nothing* — but "making a subclass reuse behavior from a parent class" matched a "what is a class?" entry via TF-IDF at a perfect 1.0 score, purely from sparse literal overlap having nothing else to compete against. A fallback-only design never revisits a match TF-IDF was "confident" about, wrong or not. Fixed by having `HybridMatcher` always compute both matchers and let semantic search override when it disagrees with TF-IDF and clears its own confidence threshold (see `app/chatbot/hybrid_matcher.py`).
4. **The default local model was unusably slow for a chat UI — until one flag fixed it.** The `qwen3:8b` model already available locally is a "thinking" model: by default it generates a full internal chain-of-reasoning block before its actual answer, which took **2 minutes 21 seconds** for a one-paragraph question on this machine's CPU-only hardware. Pulling a smaller model to fix this failed — the network here couldn't sustain a stable download of even a 1.9 GB model. The actual fix needed no new download: Ollama's API accepts `"think": false` for reasoning models, which skips that block entirely and cut the same request to ~9.5 seconds once the model was warm (`app/chatbot/llm_client.py`).

## Limitations

- The knowledge base only covers Python — 54 entries across 13 topics. Even with semantic search, matching is limited by a small general-purpose embedding model, so some ambiguous paraphrases resolve to a plausible-but-not-ideal entry rather than the perfect match. This is an inherent limitation of the approach at this scale, not a bug to chase indefinitely.
- Every chat request runs the embedding model in addition to TF-IDF (since `HybridMatcher` always computes both — see [Architecture](#architecture)), adding roughly 50–150ms of CPU-bound latency per request for KB matches. Still well within interactive chat speed.
- **AI-generated answers can be wrong.** Everything outside the Python KB goes through a local LLM, which can hallucinate like any LLM. The `source: "ai"` tag and the frontend's "AI generated" badge exist specifically so this is never confused with a verified answer — but the content itself isn't fact-checked.
- **The AI fallback needs Ollama running locally**, with the configured model (`qwen3:8b`, ~5.2 GB) already pulled. Without it, questions outside the KB get the honest static fallback message instead — the app doesn't try to auto-install or auto-pull anything.
- **AI-generated answers are much slower than KB answers** — roughly 5-15 seconds once the model is warm in memory, longer (up to the ~45s configured timeout) on the first request after Ollama's model unloads from idling. KB matches stay near-instant regardless.
- Chat history is per-browser-tab (`sessionStorage`) — it resets when the tab closes and isn't shared across devices or persisted server-side.
- No authentication or rate limiting — not intended for public unauthenticated deployment as-is.
- No knowledge-base editing UI yet — updates mean hand-editing `data/knowledge_base.json`.

## Future improvements

- **Admin UI for knowledge-base management** — authenticated CRUD endpoints plus a small admin page to add/edit/remove Q&A entries without hand-editing JSON.
- **Persistent chat history** — an optional SQLite-backed history endpoint for users who want their conversations to survive across sessions/devices.
- **Expanded knowledge base** — more entries per topic, which would also sharpen semantic matching, and would let more questions be answered by the verified path instead of the AI fallback.
- **Streaming the AI response** — the LLM path currently waits for the full generation before responding; streaming tokens back (Ollama supports this) would make the 5-15s wait feel much shorter.
- **Multi-turn follow-up questions** — e.g. "what about tuples?" after a list question, which needs conversation context the current stateless matcher (and the stateless LLM calls) don't keep.
- **Graceful UI detection of a missing Ollama install** — right now an unreachable Ollama just produces the static fallback message; the frontend could detect this via `/api/health` and hide/relabel the AI capability instead of silently degrading.
- **Deployment** — containerize with Docker and add a production ASGI setup (e.g. Uvicorn + Gunicorn workers) behind HTTPS; note that a Docker deployment would need Ollama reachable from the container (e.g. `host.docker.internal`), not just `localhost`.

## Screenshots

Not committed to the repo (to keep it lightweight), but easy to reproduce in under a minute:

1. `uvicorn app.main:app --reload`
2. Open `http://127.0.0.1:8000`
3. Click a suggested question, or ask your own — the resulting bubbles (with category tag, answer, and a copyable code block) make a good desktop screenshot.
4. In Chrome/Edge DevTools, toggle device toolbar (`Ctrl+Shift+M`) and pick a phone preset for a mobile screenshot.

## License

Personal portfolio project — no license file included. Feel free to ask before reusing substantial portions.
