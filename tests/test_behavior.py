"""
Unit tests for Stage T021: Behavioral Pattern Extraction.
Fast synthetic unit tests (< 1s) validating behavioral dimensions,
range bounds [0.0, 1.0], confidence scores, and exemplar extraction.
"""

from __future__ import annotations

import pytest

from src.nlp.behavior import BehavioralExtractor, BehavioralProfile


@pytest.fixture
def extractor() -> BehavioralExtractor:
    return BehavioralExtractor()


def test_behavior_dimensions_range(extractor):
    sample_pairs = [
        {"pair_id": "p1", "target_text": "ha pakka done bhai", "context_text": "done?"},
        {"pair_id": "p2", "target_text": "nahi re bilkul galat bolra", "context_text": "sahi hai na"},
        {"pair_id": "p3", "target_text": "api deploy karke test kia curl se", "context_text": "update?"},
        {"pair_id": "p4", "target_text": "bc lol 😭😭", "context_text": "ye dekh"},
        {"pair_id": "p5", "target_text": "shayad kal ho sakta hai i think", "context_text": "kab hoga?"},
    ]

    profile = extractor.extract_profile(sample_pairs, name="test_corpus", max_exemplars=3)

    assert isinstance(profile, BehavioralProfile)
    assert profile.total_turns == 5

    expected_dims = {
        "directness",
        "verbosity",
        "hedging",
        "disagreement_style",
        "humor_playfulness",
        "question_initiative",
        "confidence_assertion",
        "emotional_expressiveness",
        "code_switching_propensity",
        "technical_depth",
    }

    assert set(profile.dimensions.keys()) == expected_dims

    for dim_name, dim in profile.dimensions.items():
        assert 0.0 <= dim.score <= 1.0, f"Dimension {dim_name} score {dim.score} out of bounds"
        assert 0.0 <= dim.confidence <= 1.0, f"Dimension {dim_name} confidence {dim.confidence} out of bounds"
        assert isinstance(dim.supporting_exemplars, list)


def test_supporting_exemplars_mined(extractor):
    sample_pairs = [
        {"pair_id": "p_tech", "target_text": "api endpoint payload curl server backend", "context_text": "check"},
        {"pair_id": "p_humor", "target_text": "bc lol 😭😭😂", "context_text": "kya hua"},
    ]

    profile = extractor.extract_profile(sample_pairs, name="test_corpus")

    tech_dim = profile.dimensions["technical_depth"]
    assert len(tech_dim.supporting_exemplars) > 0
    assert tech_dim.supporting_exemplars[0]["pair_id"] == "p_tech"

    humor_dim = profile.dimensions["humor_playfulness"]
    assert len(humor_dim.supporting_exemplars) > 0
    assert humor_dim.supporting_exemplars[0]["pair_id"] == "p_humor"


def test_empty_input_raises_error(extractor):
    with pytest.raises(ValueError, match="cannot be empty"):
        extractor.extract_profile([])
