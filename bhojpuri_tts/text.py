import re
import unicodedata

from f5_tts.model.utils import convert_char_to_pinyin

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"))
_WHITESPACE = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[।॥.!?])\s+")


def normalize_text(text: str) -> str:
    # The char tokenizer maps unknown code points to space, so NFC matters: visually identical
    # Devanagari strings can otherwise produce different token IDs.
    text = unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)
    return _WHITESPACE.sub(" ", text).strip()


def to_char_tokens(texts: list[str]) -> list[list[str]]:
    # Same conversion IndicF5 applies at inference, so training and inference see identical tokens.
    return convert_char_to_pinyin(texts)


def split_sentences(text: str, max_chars: int) -> list[str]:
    chunks, current = [], ""
    for sentence in _SENTENCE_END.split(text):
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks
