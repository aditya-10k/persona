"""
Unit tests for Stage T018: Code-Switching & Language Analyzer.
Fast synthetic unit tests (< 1 second) testing token-level language identification,
homograph disambiguation, turn-level mode classification, switch points, and style breakdowns.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.nlp.code_switching import CodeSwitchingAnalyzer, LanguageProfile, TurnLanguageResult


@pytest.fixture
def analyzer() -> CodeSwitchingAnalyzer:
    return CodeSwitchingAnalyzer()


def test_classify_tokens_basic(analyzer):
    # Pure English tokens
    tokens_eng = ["we", "deployed", "the", "backend", "server"]
    tags_eng = analyzer.classify_tokens(tokens_eng)
    assert all(tag == "ENG" for tag in tags_eng)

    # Pure Hindi tokens
    tokens_hin = ["bhai", "kal", "milte", "hai", "sham", "ko"]
    tags_hin = analyzer.classify_tokens(tokens_hin)
    assert all(tag == "HIN" for tag in tags_hin)


def test_homograph_disambiguation(analyzer):
    # 'me': English pronoun in English sentence
    res_eng = analyzer.analyze_turn("call me on phone")
    tag_me_eng = next(tag for tok, tag in res_eng.token_tags if tok == "me")
    assert tag_me_eng == "ENG"

    # 'me': Hindi locative postposition in Hindi sentence
    res_hin = analyzer.analyze_turn("room me baitha hu bhai")
    tag_me_hin = next(tag for tok, tag in res_hin.token_tags if tok == "me")
    assert tag_me_hin == "HIN"

    # 'he': English pronoun in English sentence
    res_he_eng = analyzer.analyze_turn("he is working now")
    tag_he_eng = next(tag for tok, tag in res_he_eng.token_tags if tok == "he")
    assert tag_he_eng == "ENG"

    # 'he': Hindi copula in Hindi sentence
    res_he_hin = analyzer.analyze_turn("wo sahi bolra he")
    tag_he_hin = next(tag for tok, tag in res_he_hin.token_tags if tok == "he")
    assert tag_he_hin == "HIN"


def test_analyze_turn_modes_and_switch_points(analyzer):
    # 1. Pure English
    res_pure_eng = analyzer.analyze_turn("works for us now definitely")
    assert res_pure_eng.classification == "pure_english"
    assert res_pure_eng.eng_ratio == 1.0
    assert res_pure_eng.hin_ratio == 0.0
    assert res_pure_eng.switch_points == 0

    # 2. Pure Hindi
    res_pure_hin = analyzer.analyze_turn("bhai kal milte hai sham ko")
    assert res_pure_hin.classification == "pure_hindi"
    assert res_pure_hin.hin_ratio == 1.0
    assert res_pure_hin.eng_ratio == 0.0
    assert res_pure_hin.switch_points == 0

    # 3. Code-switched Hinglish
    res_switched = analyzer.analyze_turn("angular me kaam kia hena? ya ts online aa bhai")
    assert res_switched.classification == "code_switched"
    assert res_switched.eng_ratio > 0.0
    assert res_switched.hin_ratio > 0.0
    assert res_switched.switch_points >= 2


def test_fit_corpus_language_profile(analyzer):
    sample_pairs = [
        {"pair_id": "p1", "target_text": "works for us now"},
        {"pair_id": "p2", "target_text": "bhai kal milte hai sham ko"},
        {"pair_id": "p3", "target_text": "backend api test kar liya mene"},
        {"pair_id": "p4", "target_text": "💀💀"},
    ]

    profile = analyzer.fit(sample_pairs, name="test_corpus")

    assert isinstance(profile, LanguageProfile)
    assert profile.total_turns == 4

    # Token distribution
    tok_dist = profile.token_distribution
    assert tok_dist["english_token_ratio"] > 0
    assert tok_dist["hindi_token_ratio"] > 0
    assert pytest.approx(tok_dist["english_token_ratio"] + tok_dist["hindi_token_ratio"], rel=1e-3) == 1.0

    # Turn classification
    turn_dist = profile.turn_classification
    assert turn_dist["pure_english_ratio"] == 0.25  # p1
    assert turn_dist["pure_hindi_ratio"] == 0.25    # p2
    assert turn_dist["code_switched_ratio"] == 0.25  # p3
    assert turn_dist["neutral_ratio"] == 0.25        # p4

    # Borrowed English words
    borrowed_words = [b["word"] for b in profile.borrowed_english_words]
    assert "backend" in borrowed_words or "api" in borrowed_words or "test" in borrowed_words


def test_analyze_by_style(analyzer):
    sample_pairs = [
        {"pair_id": "p1", "target_text": "curl endpoint reservation api payload json schema"},
        {"pair_id": "p2", "target_text": "angular dev server update test"},
        {"pair_id": "p3", "target_text": "bhai bohot thak gaya hu aaj sach me"},
        {"pair_id": "p4", "target_text": "kuch nahi hua re chodd de"},
    ]
    style_labels = np.array([4, 4, 1, 1], dtype=int)
    style_names = {4: "Technical_Collab", 1: "Venting_Banter"}

    profiles = analyzer.analyze_by_style(sample_pairs, style_labels, style_names)

    assert "Technical_Collab" in profiles
    assert "Venting_Banter" in profiles

    tech_prof = profiles["Technical_Collab"]
    vent_prof = profiles["Venting_Banter"]

    # Technical Collab should have a vastly higher English ratio than Venting
    assert tech_prof.token_distribution["english_token_ratio"] > vent_prof.token_distribution["english_token_ratio"]
    assert vent_prof.token_distribution["hindi_token_ratio"] > tech_prof.token_distribution["hindi_token_ratio"]


def test_empty_input_raises_error(analyzer):
    with pytest.raises(ValueError, match="cannot be empty"):
        analyzer.fit([])
