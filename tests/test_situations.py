"""
Unit tests for Stage T020: Situational Classification.
Fast synthetic unit tests (< 1s) testing situational detection across
technical, academic/career, conflict, acknowledgement, and probing environments.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.nlp.situations import SituationalCategory, SituationalClassifier, SituationalProfile, SituationResult


@pytest.fixture
def classifier() -> SituationalClassifier:
    return SituationalClassifier()


def test_classify_technical_collab(classifier):
    res = classifier.classify_turn(
        text="curl payload update karke dev server pe deploy kia",
        context_text="api endpoint test karlo",
    )
    assert res.primary_situation == SituationalCategory.TECHNICAL_COLLAB.value
    assert res.confidence > 0.3
    assert len(res.cues_detected[SituationalCategory.TECHNICAL_COLLAB.value]) > 0


def test_classify_career_academic(classifier):
    res = classifier.classify_turn(
        text="placement company ka oa test paper kaisa gaya?",
        context_text="round 2 interview clear hua",
    )
    assert res.primary_situation == SituationalCategory.CAREER_ACADEMIC.value
    assert len(res.cues_detected[SituationalCategory.CAREER_ACADEMIC.value]) > 0


def test_classify_conflict_friction(classifier):
    res = classifier.classify_turn(
        text="nahi re bilkul galat bakchodi hai ye",
        context_text="ye karle na",
    )
    assert res.primary_situation == SituationalCategory.CONFLICT_FRICTION.value


def test_classify_acknowledgement(classifier):
    res_ha = classifier.classify_turn("ha")
    assert res_ha.primary_situation == SituationalCategory.ACKNOWLEDGEMENT.value

    res_ok = classifier.classify_turn("ok")
    assert res_ok.primary_situation == SituationalCategory.ACKNOWLEDGEMENT.value

    res_done = classifier.classify_turn("done")
    assert res_done.primary_situation == SituationalCategory.ACKNOWLEDGEMENT.value


def test_classify_advice_probing(classifier):
    res = classifier.classify_turn(
        text="kaise karna hai bhai btao?",
        context_text="check karlo",
    )
    assert res.primary_situation == SituationalCategory.ADVICE_PROBING.value


def test_fit_situational_profile(classifier):
    sample_pairs = [
        {"pair_id": "p1", "target_text": "api endpoint curl deploy", "context_text": "server test"},
        {"pair_id": "p2", "target_text": "placement interview company round", "context_text": "package"},
        {"pair_id": "p3", "target_text": "nahi galat hai", "context_text": "aisa kar"},
        {"pair_id": "p4", "target_text": "ha", "context_text": "done?"},
    ]

    profile = classifier.fit(sample_pairs, name="test_corpus")
    assert isinstance(profile, SituationalProfile)
    assert profile.total_turns == 4
    assert profile.situation_distribution[SituationalCategory.TECHNICAL_COLLAB.value] > 0
    assert profile.situation_distribution[SituationalCategory.CAREER_ACADEMIC.value] > 0
    assert profile.situation_distribution[SituationalCategory.CONFLICT_FRICTION.value] > 0
    assert profile.situation_distribution[SituationalCategory.ACKNOWLEDGEMENT.value] > 0


def test_analyze_by_style(classifier):
    sample_pairs = [
        {"pair_id": "p1", "target_text": "api test branch code", "context_text": "server"},
        {"pair_id": "p2", "target_text": "ha", "context_text": "ok"},
    ]
    style_labels = np.array([4, 5], dtype=int)
    style_names = {4: "Tech", 5: "Confirm"}

    profiles = classifier.analyze_by_style(sample_pairs, style_labels, style_names)
    assert "Tech" in profiles
    assert "Confirm" in profiles
    assert profiles["Tech"].situation_distribution[SituationalCategory.TECHNICAL_COLLAB.value] > 0.5
    assert profiles["Confirm"].situation_distribution[SituationalCategory.ACKNOWLEDGEMENT.value] > 0.5


def test_empty_input_raises_error(classifier):
    with pytest.raises(ValueError, match="cannot be empty"):
        classifier.fit([])
