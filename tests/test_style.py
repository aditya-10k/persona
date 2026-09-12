"""
Unit tests for Stages T022 & T023: Global Style Profile & Situational Style Profiles.
Fast synthetic unit tests (< 1s) validating profile generation, dimension bounds,
surface constraints, situation modulation, and exemplar extraction.
"""

from __future__ import annotations

import pytest

from src.persona.style import GlobalStyleProfile, SituationalStyleProfile, StyleProfileBuilder


@pytest.fixture
def builder() -> StyleProfileBuilder:
    return StyleProfileBuilder()


def test_build_global_style_profile(builder):
    sample_pairs = [
        {"pair_id": "p1", "target_text": "api endpoint curl deploy test", "context_text": "status check"},
        {"pair_id": "p2", "target_text": "ha pakka done", "context_text": "ho gaya?"},
        {"pair_id": "p3", "target_text": "nahi re bilkul galat", "context_text": "aisa hai"},
        {"pair_id": "p4", "target_text": "bc lol 😭😭", "context_text": "dekha ye"},
    ]

    profile = builder.build_global_style(sample_pairs, name="test_global")

    assert isinstance(profile, GlobalStyleProfile)
    assert profile.total_turns == 4

    # Core dimensions
    dims = profile.core_dimensions
    expected_dims = {
        "directness",
        "formality",
        "verbosity",
        "humor",
        "sarcasm",
        "hedging",
        "emotional_expression",
        "question_initiative",
        "confidence",
        "code_switching_propensity",
        "technical_depth",
    }
    assert set(dims.keys()) == expected_dims

    for dim_name, score in dims.items():
        assert 0.0 <= score <= 1.0, f"Dimension {dim_name} score {score} out of bounds"

    # Surface constraints
    surf = profile.surface_constraints
    assert "zero_terminal_punctuation_ratio" in surf
    assert "all_lowercase_ratio" in surf
    assert "mean_bubbles_per_turn" in surf

    # Syntactic constraints
    syn = profile.syntactic_constraints
    assert "verbless_fragment_ratio" in syn
    assert "simple_clause_ratio" in syn

    # Language constraints
    lang = profile.language_constraints
    assert "english_token_ratio" in lang
    assert "hindi_token_ratio" in lang

    # Discourse moves
    assert "act_distribution" in profile.discourse_moves


def test_build_situational_style_profiles(builder):
    sample_pairs = [
        {"pair_id": "p1", "target_text": "api deploy server test curl payload", "context_text": "backend"},
        {"pair_id": "p2", "target_text": "ha", "context_text": "ok?"},
        {"pair_id": "p3", "target_text": "nahi re galat", "context_text": "ye karle"},
    ]

    profiles = builder.build_situational_styles(sample_pairs, max_exemplars=2)

    assert isinstance(profiles, dict)
    assert len(profiles) > 0

    for sit_key, prof in profiles.items():
        assert isinstance(prof, SituationalStyleProfile)
        assert prof.total_turns > 0
        assert 0.0 < prof.situation_prevalence <= 1.0
        assert "directness" in prof.core_dimensions
        assert "zero_terminal_punctuation_ratio" in prof.surface_constraints
        assert "english_token_ratio" in prof.language_constraints
        assert isinstance(prof.representative_exemplars, list)


def test_empty_input_raises_error(builder):
    with pytest.raises(ValueError, match="cannot be empty"):
        builder.build_global_style([])

    with pytest.raises(ValueError, match="cannot be empty"):
        builder.build_situational_styles([])
