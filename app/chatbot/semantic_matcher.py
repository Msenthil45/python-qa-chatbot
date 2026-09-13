from typing import List

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.chatbot.knowledge_base import KnowledgeBase
from app.chatbot.match_result import MatchResult
from app.config import SEMANTIC_MODEL_NAME, SEMANTIC_SIMILARITY_THRESHOLD


class SemanticMatcher:
    """
    Finds the closest knowledge base entry to a user's question using
    sentence embeddings and cosine similarity, instead of TF-IDF's literal
    word overlap. This catches paraphrases — e.g. "how can a function
    accept a variable number of inputs?" for a *args/**kwargs entry — that
    share few or no exact words with anything in the knowledge base.
    Below `threshold`, `match()` reports no match rather than guessing.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
        model_name: str = SEMANTIC_MODEL_NAME,
    ):
        corpus: List[tuple] = knowledge_base.searchable_corpus()
        if not corpus:
            raise ValueError("Cannot build a matcher from an empty knowledge base")

        self._kb = knowledge_base
        self._threshold = threshold
        self._corpus_entry_indices = [idx for idx, _ in corpus]

        corpus_texts = [text for _, text in corpus]
        self._model = SentenceTransformer(model_name)
        self._corpus_embeddings = self._model.encode(corpus_texts, normalize_embeddings=True)

    def match(self, user_question: str) -> MatchResult:
        cleaned = user_question.strip()
        if not cleaned:
            return MatchResult(matched=False, score=0.0, entry=None)

        query_embedding = self._model.encode([cleaned], normalize_embeddings=True)
        similarities = cosine_similarity(query_embedding, self._corpus_embeddings)[0]

        best_position = similarities.argmax()
        best_score = float(similarities[best_position])
        best_entry_index = self._corpus_entry_indices[best_position]

        if best_score >= self._threshold:
            return MatchResult(
                matched=True,
                score=best_score,
                entry=self._kb.get_by_index(best_entry_index),
            )
        return MatchResult(matched=False, score=best_score, entry=None)
