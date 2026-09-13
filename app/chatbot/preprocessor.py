import re

# sklearn's default TfidfVectorizer token pattern only keeps word characters,
# so symbol-only operators (no letters/digits around them) would otherwise
# vanish completely during tokenization, losing all matching signal for
# questions about them (e.g. "==" vs "is"). Spelling them out preserves it.
_OPERATOR_WORDS = [
    (re.compile(r"=="), " equals "),
    (re.compile(r"!="), " notequals "),
    (re.compile(r"<="), " lessthanorequal "),
    (re.compile(r">="), " greaterthanorequal "),
    (re.compile(r"//"), " floordivision "),
    (re.compile(r"%"), " modulus "),
]


def clean_text(text: str) -> str:
    """
    Normalization applied identically to knowledge base text and user
    questions before TF-IDF vectorization: lowercase, spell out bare
    operator symbols, and collapse whitespace. Deliberately avoids
    stripping general punctuation — sklearn's TfidfVectorizer tokenizer
    already ignores it, and stripping it here first would also mangle
    meaningful tokens like "__init__".
    """
    text = text.strip().lower()
    for pattern, replacement in _OPERATOR_WORDS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
