from typing import List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.chatbot.knowledge_base import KnowledgeBase
from app.chatbot.match_result import MatchResult
from app.chatbot.preprocessor import clean_text
from app.config import SIMILARITY_THRESHOLD

# "python" appears in almost every knowledge base entry, so it carries no
# discriminating power for this domain — treat it as a stop word alongside
# sklearn's standard English list, on top of which it would otherwise let
# a single generic word like "python" tie-match many unrelated entries.
_STOP_WORDS = list(ENGLISH_STOP_WORDS.union({"python"}))


class QuestionMatcher:
    """
    Finds the closest knowledge base entry to a user's question using
    TF-IDF vectorization and cosine similarity. Never fabricates an
    answer: below `threshold`, `match()` reports no match so the caller
    can fall back instead of guessing.
    """

    def __init__(self, knowledge_base: KnowledgeBase, threshold: float = SIMILARITY_THRESHOLD):
        corpus: List[tuple] = knowledge_base.searchable_corpus()
        if not corpus:
            raise ValueError("Cannot build a matcher from an empty knowledge base")

        self._kb = knowledge_base
        self._threshold = threshold
        self._corpus_entry_indices = [idx for idx, _ in corpus]

        corpus_texts = [clean_text(text) for _, text in corpus]
        # Stop-word removal is essential here: without it, generic words
        # shared by almost every question ("what", "is", "how", "do") inflate
        # similarity scores enough to make unrelated questions (e.g. "how do
        # i cook pasta") falsely cross the confidence threshold.
        self._vectorizer = TfidfVectorizer(stop_words=_STOP_WORDS, ngram_range=(1, 2))
        self._corpus_matrix = self._vectorizer.fit_transform(corpus_texts)

    def match(self, user_question: str) -> MatchResult:
        cleaned = clean_text(user_question)
        if not cleaned:
            return MatchResult(matched=False, score=0.0, entry=None)

        user_vector = self._vectorizer.transform([cleaned])
        similarities = cosine_similarity(user_vector, self._corpus_matrix)[0]

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
