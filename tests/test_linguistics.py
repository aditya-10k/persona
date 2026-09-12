"""
Unit tests for Stage T016: Surface Linguistic Profiler.
Fast synthetic unit tests (< 1 second) testing turn feature extraction,
lexical diversity indices, casing, punctuation, emojis, and style breakdown.
"""

from __future__ import annotations

import pytest

from src.nlp.linguistics import LinguisticProfile, LinguisticProfiler


@pytest.fixture
def profiler() -> LinguisticProfiler:
    return LinguisticProfiler()


def test_analyze_turn_casing(profiler):
    # All lowercase
    res_lower = profiler.analyze_turn("bhai kal milte hai sham ko")
    assert res_lower["is_all_lower"] is True
    assert res_lower["is_all_upper"] is False
    assert res_lower["has_initial_cap"] is False

    # All uppercase
    res_upper = profiler.analyze_turn("KYA HUA BHAI")
    assert res_upper["is_all_upper"] is True
    assert res_upper["is_all_lower"] is False

    # Capitalized start
    res_cap = profiler.analyze_turn("Kal chalte hai")
    assert res_cap["has_initial_cap"] is True
    assert res_cap["is_all_lower"] is False


def test_analyze_turn_punctuation(profiler):
    # Zero terminal punctuation
    res_zero = profiler.analyze_turn("aaj code deploy kar diya")
    assert res_zero["has_terminal_punct"] is False

    # Terminal punctuation
    res_punct = profiler.analyze_turn("aaj code deploy kar diya.")
    assert res_punct["has_terminal_punct"] is True

    # Multi-question
    res_q = profiler.analyze_turn("kya hua???")
    assert res_q["has_question"] is True
    assert res_q["has_multi_question"] is True

    # Ellipsis
    res_ell = profiler.analyze_turn("dekhte hai...")
    assert res_ell["has_ellipsis"] is True


def test_analyze_turn_emojis(profiler):
    # Pure emoji burst
    res_em_burst = profiler.analyze_turn("😂😂😂")
    assert res_em_burst["emoji_count"] == 3
    assert res_em_burst["is_emoji_only"] is True
    assert res_em_burst["has_emoji_burst"] is True

    # Mixed text + emoji
    res_mixed = profiler.analyze_turn("mast joke tha bhai 💀")
    assert res_mixed["emoji_count"] == 1
    assert res_mixed["is_emoji_only"] is False
    assert res_mixed["has_emoji_burst"] is False
    assert res_mixed["emojis"] == ["💀"]


def test_analyze_turn_elongations_and_abbreviations(profiler):
    text = "soooo cool clg kab chalna he grp pe btao"
    res = profiler.analyze_turn(text)

    assert res["has_elongation"] is True
    assert "soooo" in res["elongations"]
    assert "clg" in res["abbreviations"]
    assert "grp" in res["abbreviations"]


def test_fit_corpus_profile(profiler):
    sample_pairs = [
        {
            "pair_id": "p001",
            "target_text": "bhai kal milte hai sham ko",
            "target_response": {"message_count": 1},
        },
        {
            "pair_id": "p002",
            "target_text": "nahi yaar bohot thak gaya hu aaj...",
            "target_response": {"message_count": 2},
        },
        {
            "pair_id": "p003",
            "target_text": "KYA BAAT KAR RAHA HAI???",
            "target_response": {"message_count": 1},
        },
        {
            "pair_id": "p004",
            "target_text": "💀💀",
            "target_response": {"message_count": 1},
        },
        {
            "pair_id": "p005",
            "target_text": "haana sahi bola",
            "target_response": {"message_count": 3},
        },
    ]

    prof = profiler.fit(sample_pairs, name="test_corpus")

    assert isinstance(prof, LinguisticProfile)
    assert prof.total_turns == 5
    assert prof.total_messages == 8  # 1 + 2 + 1 + 1 + 3

    # Volumetrics
    assert prof.words_per_turn["mean"] > 0
    assert prof.chars_per_turn["mean"] > 0

    # Lexical Diversity
    lex = prof.lexical_diversity
    assert 0.0 < lex["ttr"] <= 1.0
    assert lex["root_ttr_guiraud"] > 0
    assert lex["hapax_legomena_count"] > 0
    assert 0.0 < lex["hapax_ratio"] <= 1.0

    # Burstiness
    burst = prof.burstiness
    assert burst["single_bubble_ratio"] == 0.6  # 3 out of 5 turns
    assert burst["double_bubble_ratio"] == 0.2  # 1 out of 5
    assert burst["multi_bubble_ratio"] == 0.2   # 1 out of 5

    # Casing
    casing = prof.casing_distribution
    assert casing["all_lowercase_ratio"] == 0.6  # p001, p002, p005
    assert casing["all_uppercase_ratio"] == 0.2  # p003

    # Punctuation
    punct = prof.punctuation_metrics
    assert punct["question_turns_ratio"] == 0.2
    assert punct["multi_question_ratio"] == 0.2
    assert punct["ellipsis_turns_ratio"] == 0.2

    # Emojis
    em = prof.emoji_fingerprint
    assert em["turns_with_emojis_ratio"] == 0.2  # p004
    assert em["emoji_only_turns_ratio"] == 0.2


def test_analyze_by_style(profiler):
    import numpy as np

    sample_pairs = [
        {"pair_id": "p01", "target_text": "💀💀", "target_response": {"message_count": 1}},
        {"pair_id": "p02", "target_text": "Padh ya soja", "target_response": {"message_count": 1}},
        {"pair_id": "p03", "target_text": "angular me payload update kia he", "target_response": {"message_count": 2}},
        {"pair_id": "p04", "target_text": "backend api curl check kar", "target_response": {"message_count": 1}},
    ]
    style_labels = np.array([2, 2, 4, 4], dtype=int)
    style_names = {2: "Reactive_Slang", 4: "Technical_Collab"}

    style_profiles = profiler.analyze_by_style(sample_pairs, style_labels, style_names)

    assert "Reactive_Slang" in style_profiles
    assert "Technical_Collab" in style_profiles

    reactive_prof = style_profiles["Reactive_Slang"]
    tech_prof = style_profiles["Technical_Collab"]

    assert reactive_prof.total_turns == 2
    assert tech_prof.total_turns == 2
    # Tech words mean should be higher than reactive words mean
    assert tech_prof.words_per_turn["mean"] > reactive_prof.words_per_turn["mean"]


def test_empty_pairs_raise_error(profiler):
    with pytest.raises(ValueError, match="cannot be empty"):
        profiler.fit([])
