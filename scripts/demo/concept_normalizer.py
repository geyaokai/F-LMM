from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

try:
    import spacy

    _spacy_nlp = None
    _spacy_warned = False
except ImportError:
    spacy = None
    _spacy_nlp = None
    _spacy_warned = False

_CONCEPT_GENERIC_WORDS = {
    "answer",
    "area",
    "background",
    "color",
    "environment",
    "everything",
    "hour",
    "hours",
    "image",
    "item",
    "kind",
    "object",
    "one",
    "ones",
    "pair",
    "part",
    "photo",
    "picture",
    "piece",
    "question",
    "scene",
    "set",
    "side",
    "something",
    "sort",
    "style",
    "stuff",
    "setting",
    "thing",
    "type",
    "view",
}

_CONCEPT_ATTRIBUTE_WORDS = {
    "above",
    "behind",
    "below",
    "big",
    "black",
    "blue",
    "bottom",
    "brown",
    "center",
    "central",
    "daylight",
    "front",
    "gray",
    "green",
    "grey",
    "large",
    "left",
    "little",
    "middle",
    "orange",
    "pink",
    "purple",
    "red",
    "right",
    "sideways",
    "silver",
    "small",
    "top",
    "upper",
    "white",
    "yellow",
}

_CONCEPT_COMPOUND_DEPS = {"compound", "flat", "fixed", "goeswith"}


@dataclass(frozen=True)
class _RawVisualPhrase:
    mention_text: str
    mention_char_span: Tuple[int, int]
    concept_text: Optional[str]
    concept_char_span: Optional[Tuple[int, int]]
    concept_lemma: Optional[str]
    concept_source: Optional[str]


def get_spacy_nlp():
    global _spacy_nlp, _spacy_warned
    if _spacy_nlp is not None:
        return _spacy_nlp
    if spacy is None:
        if not _spacy_warned:
            print(
                "[Debug] spaCy import unavailable; phrase extraction will use model/regex fallback."
            )
            _spacy_warned = True
        return None
    try:
        _spacy_nlp = spacy.load("en_core_web_sm")
    except Exception:
        if not _spacy_warned:
            print(
                "[Debug] spaCy model 'en_core_web_sm' unavailable; phrase extraction will use model/regex fallback."
            )
            _spacy_warned = True
        _spacy_nlp = None
    return _spacy_nlp


def _normalize_concept_key(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _is_filtered_concept(text: str, lemma: Optional[str] = None) -> bool:
    key = _normalize_concept_key(text)
    lemma_key = _normalize_concept_key(lemma or "")
    if not key:
        return True
    if key in _CONCEPT_GENERIC_WORDS or lemma_key in _CONCEPT_GENERIC_WORDS:
        return True
    words = key.split()
    if words and all(word in _CONCEPT_ATTRIBUTE_WORDS for word in words):
        return True
    return False


def _char_to_token(offsets: Sequence[Tuple[int, int]], char_pos: int) -> int:
    for idx, (start, end) in enumerate(offsets):
        if start <= char_pos < end:
            return idx
    return len(offsets) - 1


def _clip_char_span(
    answer_text: str, char_span: Sequence[int]
) -> Optional[Tuple[int, int]]:
    if not answer_text or not char_span or len(char_span) < 2:
        return None
    try:
        start, end = int(char_span[0]), int(char_span[1])
    except Exception:
        return None
    max_len = len(answer_text)
    start = max(0, min(start, max_len))
    end = max(0, min(end, max_len))
    if end <= start:
        return None
    return start, end


def _concept_token_span_from_char_span(
    offsets: Sequence[Tuple[int, int]],
    char_span: Optional[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    if not offsets or not char_span:
        return None
    start, end = char_span
    if end <= start:
        return None
    token_start = _char_to_token(offsets, start)
    token_end = _char_to_token(offsets, max(start, end - 1)) + 1
    if token_end <= token_start:
        token_end = token_start + 1
    return token_start, token_end


def _extract_visual_concept_spacy(
    mention_text: str,
) -> Optional[Tuple[str, Tuple[int, int], str, str]]:
    nlp = get_spacy_nlp()
    if nlp is None:
        return None
    doc = nlp(mention_text)
    noun_tokens = [
        token
        for token in doc
        if token.pos_ in {"NOUN", "PROPN"} and token.text.strip() and token.is_alpha
    ]
    if not noun_tokens:
        return None

    head = None
    for token in reversed(noun_tokens):
        lemma = token.lemma_ if token.lemma_ and token.lemma_ != "-PRON-" else token.text
        if not _is_filtered_concept(token.text, lemma):
            head = token
            break
    if head is None:
        head = noun_tokens[-1]

    concept_indices = {head.i}
    changed = True
    while changed:
        changed = False
        for token in doc:
            if token.i in concept_indices:
                continue
            if token.dep_ not in _CONCEPT_COMPOUND_DEPS:
                continue
            if token.head.i not in concept_indices:
                continue
            concept_indices.add(token.i)
            changed = True

    keep = [doc[idx] for idx in sorted(concept_indices)]
    start = min(token.idx for token in keep)
    end = max(token.idx + len(token.text) for token in keep)
    concept_text = mention_text[start:end].strip()
    if not concept_text:
        return None
    lemma_parts = []
    for token in keep:
        raw_lemma = token.lemma_ if token.lemma_ and token.lemma_ != "-PRON-" else token.text
        lemma_parts.append(raw_lemma.lower())
    concept_lemma = " ".join(part for part in lemma_parts if part).strip() or concept_text.lower()
    if _is_filtered_concept(concept_text, concept_lemma):
        return None
    source = "spacy_compound" if len(keep) > 1 else "spacy_head"
    return concept_text, (start, end), concept_lemma, source


def _extract_visual_concept_heuristic(
    mention_text: str,
) -> Optional[Tuple[str, Tuple[int, int], str, str]]:
    matches = list(re.finditer(r"[A-Za-z]+(?:['-][A-Za-z]+)?", mention_text))
    if not matches:
        return None
    for match in reversed(matches):
        token_text = match.group(0).strip()
        token_key = token_text.lower().strip("'")
        if not token_key:
            continue
        if token_key.endswith("'s"):
            continue
        if _is_filtered_concept(token_text, token_key):
            continue
        return token_text, (match.start(), match.end()), token_key, "heuristic_head"
    return None


def _derive_visual_concept_chars(
    answer_text: str,
    mention_text: str,
    mention_char_span: Tuple[int, int],
) -> Tuple[
    Optional[str],
    Optional[Tuple[int, int]],
    Optional[str],
    Optional[str],
]:
    if not mention_text:
        return None, None, None, "empty"

    concept = _extract_visual_concept_spacy(mention_text)
    if concept is None:
        concept = _extract_visual_concept_heuristic(mention_text)
    if concept is None:
        return None, None, None, "unresolved"

    concept_text, local_char_span, concept_lemma, concept_source = concept
    global_char_span = (
        int(mention_char_span[0] + local_char_span[0]),
        int(mention_char_span[0] + local_char_span[1]),
    )
    clipped_char_span = _clip_char_span(answer_text, global_char_span)
    if clipped_char_span is None:
        return None, None, None, "char_span_invalid"
    return concept_text, clipped_char_span, concept_lemma, concept_source


def derive_visual_concept(
    answer_text: str,
    mention_text: str,
    mention_char_span: Tuple[int, int],
    offsets: Sequence[Tuple[int, int]],
) -> Tuple[
    Optional[str],
    Optional[Tuple[int, int]],
    Optional[Tuple[int, int]],
    Optional[str],
    Optional[str],
]:
    (
        concept_text,
        concept_char_span,
        concept_lemma,
        concept_source,
    ) = _derive_visual_concept_chars(answer_text, mention_text, mention_char_span)
    if concept_char_span is None:
        return None, None, None, concept_lemma, concept_source
    concept_token_span = _concept_token_span_from_char_span(offsets, concept_char_span)
    return (
        concept_text,
        concept_char_span,
        concept_token_span,
        concept_lemma,
        concept_source,
    )


def extract_spacy_visual_phrases(
    answer_text: str, limit: int
) -> List[Tuple[str, Tuple[int, int]]]:
    nlp = get_spacy_nlp()
    if nlp is None:
        return []
    doc = nlp(answer_text)
    raw_candidates: List[_RawVisualPhrase] = []
    seen_mentions: set[Tuple[int, int, str]] = set()

    def add_candidate(text: str, start: int, end: int):
        mention_text = text.strip()
        if not mention_text:
            return
        span = _clip_char_span(answer_text, (start, end))
        if span is None:
            return
        key = (span[0], span[1], mention_text.lower())
        if key in seen_mentions:
            return
        seen_mentions.add(key)
        (
            concept_text,
            concept_char_span,
            concept_lemma,
            concept_source,
        ) = _derive_visual_concept_chars(answer_text, mention_text, span)
        if concept_text is None:
            return
        raw_candidates.append(
            _RawVisualPhrase(
                mention_text=mention_text,
                mention_char_span=span,
                concept_text=concept_text,
                concept_char_span=concept_char_span,
                concept_lemma=concept_lemma,
                concept_source=concept_source,
            )
        )

    for chunk in doc.noun_chunks:
        add_candidate(chunk.text, chunk.start_char, chunk.end_char)

    for token in doc:
        if token.pos_ not in {"NOUN", "PROPN"}:
            continue
        if not token.text.strip():
            continue
        add_candidate(token.text, token.idx, token.idx + len(token.text))

    if not raw_candidates:
        return []

    merged_candidates = {}
    for candidate in raw_candidates:
        concept_key = (
            candidate.concept_char_span,
            _normalize_concept_key(candidate.concept_text or ""),
        )
        existing = merged_candidates.get(concept_key)
        if existing is None:
            merged_candidates[concept_key] = candidate
            continue
        existing_words = len(existing.mention_text.split())
        candidate_words = len(candidate.mention_text.split())
        if candidate_words <= 4 and (
            existing_words > 4 or len(candidate.mention_text) > len(existing.mention_text)
        ):
            merged_candidates[concept_key] = candidate
    raw_candidates = list(merged_candidates.values())

    concept_counts = {}
    for candidate in raw_candidates:
        concept_key = _normalize_concept_key(candidate.concept_text or "")
        if not concept_key:
            continue
        concept_counts[concept_key] = concept_counts.get(concept_key, 0) + 1

    final_candidates: List[Tuple[int, str, Tuple[int, int]]] = []
    seen_final: set[Tuple[int, int, str]] = set()
    for candidate in raw_candidates:
        mention_word_count = len(candidate.mention_text.split())
        concept_key = _normalize_concept_key(candidate.concept_text or "")
        use_concept = bool(candidate.concept_char_span and concept_counts.get(concept_key, 0) <= 1)
        if use_concept:
            display_text = candidate.concept_text or candidate.mention_text
            display_span = candidate.concept_char_span or candidate.mention_char_span
        else:
            display_text = candidate.mention_text
            display_span = candidate.mention_char_span
        if mention_word_count > 4 and candidate.concept_char_span:
            display_text = candidate.concept_text or display_text
            display_span = candidate.concept_char_span
        final_key = (display_span[0], display_span[1], display_text.lower())
        if final_key in seen_final:
            continue
        seen_final.add(final_key)
        final_candidates.append((display_span[0], display_text, display_span))

    final_candidates.sort(key=lambda item: item[0])
    return [(text, span) for _, text, span in final_candidates[:limit]]
