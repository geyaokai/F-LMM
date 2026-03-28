#!/usr/bin/env python3
from __future__ import annotations

import re

from scripts.demo.concept_normalizer import extract_spacy_visual_phrases, get_spacy_nlp
from scripts.demo.interact import (
    _match_selected_phrase_to_candidate,
    build_phrase_candidates,
    serialize_phrase_candidate,
)


def simple_offsets(text: str):
    return [(match.start(), match.end()) for match in re.finditer(r"[A-Za-z]+(?:['-][A-Za-z]+)?", text)]


def test_atomic_concept_from_attribute_phrase():
    answer = "A yellow taxi is parked near the curb."
    candidates = build_phrase_candidates(
        answer,
        [("yellow taxi", (-1, -1))],
        simple_offsets(answer),
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.text == "yellow taxi"
    assert candidate.concept_text == "taxi"
    assert candidate.concept_char_span is not None
    start, end = candidate.concept_char_span
    assert answer[start:end] == "taxi"
    assert candidate.concept_source in {"spacy_head", "heuristic_head"}


def test_atomic_concept_from_possessive_phrase():
    answer = "The man's shirt is yellow."
    candidates = build_phrase_candidates(
        answer,
        [("man's shirt", (-1, -1))],
        simple_offsets(answer),
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.concept_text == "shirt"
    assert candidate.concept_char_span is not None
    start, end = candidate.concept_char_span
    assert answer[start:end] == "shirt"
    assert candidate.concept_source in {"spacy_head", "heuristic_head"}


def test_compound_concept_prefers_lexicalized_noun_when_spacy_available():
    if get_spacy_nlp() is None:
        return
    answer = "A traffic light stands beside the road."
    candidates = build_phrase_candidates(
        answer,
        [("traffic light", (-1, -1))],
        simple_offsets(answer),
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.concept_text == "traffic light"
    assert candidate.concept_source == "spacy_compound"


def test_phrase_payload_contains_mention_and_concept_fields():
    answer = "A yellow taxi is parked near the curb."
    candidate = build_phrase_candidates(
        answer,
        [("yellow taxi", (-1, -1))],
        simple_offsets(answer),
    )[0]
    payload = serialize_phrase_candidate(candidate, index=3)
    assert payload["index"] == 3
    assert payload["text"] == "yellow taxi"
    assert payload["mention_text"] == "yellow taxi"
    assert payload["concept_text"] == "taxi"
    assert payload["concept_char_span"] is not None
    assert payload["concept_source"] in {"spacy_head", "heuristic_head"}


def test_spacy_phrase_extraction_prefers_visual_concepts():
    if get_spacy_nlp() is None:
        return
    answer = "The man is holding a pair of blue pants beside another yellow taxi."
    phrases = extract_spacy_visual_phrases(answer, limit=8)
    phrase_texts = [text for text, _ in phrases]
    assert "pants" in phrase_texts
    assert "a pair" not in phrase_texts
    assert any(text in {"another yellow taxi", "taxi"} for text in phrase_texts)


def test_build_phrase_candidates_filters_article_only_phrase():
    answer = "A man is standing near yellow taxis."
    candidates = build_phrase_candidates(
        answer,
        [("a", (-1, -1)), ("man", (-1, -1)), ("yellow taxis", (-1, -1))],
        simple_offsets(answer),
    )
    texts = [candidate.text for candidate in candidates]
    assert "a" not in texts
    assert "man" in texts
    assert "yellow taxis" in texts


def test_selected_phrase_matching_does_not_fallback_to_single_letter_candidate():
    candidates = [
        ("a", (-1, -1)),
        ("orange shirt", (-1, -1)),
        ("yellow taxis", (-1, -1)),
    ]
    matched = _match_selected_phrase_to_candidate("tappear taxis", candidates)
    assert matched is not None
    assert matched[0] == "yellow taxis"
