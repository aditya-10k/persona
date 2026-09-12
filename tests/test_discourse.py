"""
Unit tests for Stage T019: Discourse Analysis.
Fast synthetic unit tests (< 1s) testing conversational functions,
agreement/disagreement, question answering, causal explanations,
epistemic hedging, advice, humor, and context-response transitions.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.nlp.discourse import DiscourseAct, DiscourseAnalyzer, DiscourseProfile, TurnDiscourseResult


@pytest.fixture
def analyzer() -> DiscourseAnalyzer:
    return DiscourseAnalyzer()


def test_agreement_detection(analyzer):
    res_short = analyzer.analyze_turn("ha")
    assert res_short.primary_act == DiscourseAct.AGREE.value
    assert DiscourseAct.AGREE.value in res_short.act_tags

    res_phrase = analyzer.analyze_turn("haa sahi hai bhai definitely")
    assert res_phrase.primary_act == DiscourseAct.AGREE.value
    assert DiscourseAct.AGREE.value in res_phrase.act_tags


def test_disagreement_detection(analyzer):
    res_short = analyzer.analyze_turn("nahi re bilkul nahi")
    assert res_short.primary_act == DiscourseAct.DISAGREE.value
    assert DiscourseAct.DISAGREE.value in res_short.act_tags

    res_friction = analyzer.analyze_turn("aisa nahi hota bhai wrong hai")
    assert res_friction.primary_act == DiscourseAct.DISAGREE.value
    assert DiscourseAct.DISAGREE.value in res_friction.act_tags


def test_question_detection(analyzer):
    res_explicit = analyzer.analyze_turn("bhai link bheja kya?")
    assert res_explicit.primary_act == DiscourseAct.QUESTION.value
    assert DiscourseAct.QUESTION.value in res_explicit.act_tags

    res_wh = analyzer.analyze_turn("kya chal raha hai")
    assert res_wh.primary_act == DiscourseAct.QUESTION.value
    assert DiscourseAct.QUESTION.value in res_wh.act_tags


def test_explanation_and_clarification(analyzer):
    # Explanation
    res_exp = analyzer.analyze_turn("kyuki server down tha isliye deploy fail hua")
    assert res_exp.primary_act == DiscourseAct.EXPLAIN.value
    assert DiscourseAct.EXPLAIN.value in res_exp.act_tags

    # Clarification
    res_clarify = analyzer.analyze_turn("matlab actually front end issue tha")
    assert res_clarify.primary_act == DiscourseAct.CLARIFY.value
    assert DiscourseAct.CLARIFY.value in res_clarify.act_tags


def test_hedging_and_advice(analyzer):
    # Hedging
    res_hedge = analyzer.analyze_turn("shayad kal tak ho sakta hai i think")
    assert res_hedge.primary_act == DiscourseAct.HEDGE.value
    assert DiscourseAct.HEDGE.value in res_hedge.act_tags

    # Advice
    res_adv = analyzer.analyze_turn("ek baar restart karke check kar lo")
    assert res_adv.primary_act == DiscourseAct.ADVICE.value
    assert DiscourseAct.ADVICE.value in res_adv.act_tags


def test_humor_and_emphasis(analyzer):
    res_humor = analyzer.analyze_turn("bc lol 😭😭")
    assert res_humor.primary_act == DiscourseAct.HUMOR.value
    assert DiscourseAct.HUMOR.value in res_humor.act_tags

    res_emp = analyzer.analyze_turn("literally sach me bohot bada issue hai")
    assert DiscourseAct.EMPHASIS.value in res_emp.act_tags


def test_context_conditioned_transitions(analyzer):
    # Context was a question, response is an answer/informative
    res_ans = analyzer.analyze_turn(
        text="ha room pe hu abhi",
        context_text="kaha pe hai bhai?",
    )
    assert res_ans.context_act == DiscourseAct.QUESTION.value
    assert res_ans.primary_act in {DiscourseAct.AGREE.value, DiscourseAct.INFORMATIVE.value}


def test_fit_corpus_discourse_profile(analyzer):
    sample_pairs = [
        {"pair_id": "p1", "context": ["kaha hai?"], "target_text": "ha bhai aara hu"},
        {"pair_id": "p2", "context": ["ye sahi hai?"], "target_text": "nahi bilkul galat"},
        {"pair_id": "p3", "context": ["kyu hua aisa?"], "target_text": "kyuki port bind fail ho gaya"},
        {"pair_id": "p4", "context": ["dekh le"], "target_text": "😂😭"},
    ]

    profile = analyzer.fit(sample_pairs, name="test_corpus")

    assert isinstance(profile, DiscourseProfile)
    assert profile.total_turns == 4

    dist = profile.act_distribution
    assert dist[DiscourseAct.AGREE.value] > 0
    assert dist[DiscourseAct.DISAGREE.value] > 0
    assert dist[DiscourseAct.EXPLAIN.value] > 0
    assert dist[DiscourseAct.HUMOR.value] > 0

    # Transitions from question
    q_transitions = profile.context_response_transitions[DiscourseAct.QUESTION.value]
    assert q_transitions[DiscourseAct.AGREE.value] > 0 or q_transitions[DiscourseAct.DISAGREE.value] > 0 or q_transitions[DiscourseAct.EXPLAIN.value] > 0


def test_analyze_by_style(analyzer):
    sample_pairs = [
        {"pair_id": "p1", "target_text": "nahi bilkul nahi re"},
        {"pair_id": "p2", "target_text": "nahi re chodd de"},
        {"pair_id": "p3", "target_text": "ha pakka done bhai"},
        {"pair_id": "p4", "target_text": "ha sahi hai"},
    ]
    style_labels = np.array([0, 0, 5, 5], dtype=int)
    style_names = {0: "Denial_Friction", 5: "Minimalist_Confirm"}

    profiles = analyzer.analyze_by_style(sample_pairs, style_labels, style_names)

    assert "Denial_Friction" in profiles
    assert "Minimalist_Confirm" in profiles

    denial_prof = profiles["Denial_Friction"]
    confirm_prof = profiles["Minimalist_Confirm"]

    # Denial should have high disagree
    assert denial_prof.act_distribution[DiscourseAct.DISAGREE.value] > 0.5
    # Confirm should have high agree
    assert confirm_prof.act_distribution[DiscourseAct.AGREE.value] > 0.5


def test_empty_input_raises_error(analyzer):
    with pytest.raises(ValueError, match="cannot be empty"):
        analyzer.fit([])
