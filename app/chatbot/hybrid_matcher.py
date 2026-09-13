from app.chatbot.match_result import MatchResult
from app.chatbot.matcher import QuestionMatcher
from app.chatbot.semantic_matcher import SemanticMatcher


class HybridMatcher:
    """
    Combines the fast lexical (TF-IDF) matcher with the semantic
    (sentence-embedding) matcher, always computing both rather than
    treating semantic search as a fallback used only when TF-IDF finds
    nothing. That distinction matters: TF-IDF can return a *wrong* entry
    with deceptively high confidence (e.g. "making a subclass reuse
    behavior from a parent class" matched a "what is a class?" entry at
    a perfect 1.0 score purely from sparse literal word overlap) — a
    fallback-only design would never give semantic search a chance to
    catch that, since TF-IDF did technically find "a" match.

    Resolution rule:
      - TF-IDF found nothing            -> trust semantic (if it matched)
      - semantic isn't confident enough -> trust TF-IDF
      - both matched, same entry        -> agreement, return it
      - both matched, different entries -> trust semantic; dense
        embeddings capture meaning better than sparse word overlap in
        exactly the cases where they disagree
    """

    def __init__(self, tfidf_matcher: QuestionMatcher, semantic_matcher: SemanticMatcher):
        self._tfidf_matcher = tfidf_matcher
        self._semantic_matcher = semantic_matcher

    def match(self, user_question: str) -> MatchResult:
        tfidf_result = self._tfidf_matcher.match(user_question)
        semantic_result = self._semantic_matcher.match(user_question)

        if not tfidf_result.matched:
            return semantic_result

        if not semantic_result.matched:
            return tfidf_result

        if tfidf_result.entry["id"] == semantic_result.entry["id"]:
            return tfidf_result

        return semantic_result
