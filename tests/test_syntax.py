"""
Unit tests for Stage T017: Syntactic Profiler.
Fast synthetic unit tests (< 1 second) testing clause structures, speech acts,
imperatives, negation, conditionals, and pronoun orientation in code-mixed Hinglish.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.nlp.syntax import SyntacticProfile, SyntacticProfiler


@pytest.fixture
def profiler() -> SyntacticProfiler:
    return SyntacticProfiler()


def test_analyze_turn_clause_structure(profiler):
    # Verbless fragment
    res_frag = profiler.analyze_turn_syntax("Huh")
    assert res_frag["clause_type"] == "verbless_fragment"

    res_frag2 = profiler.analyze_turn_syntax("Nahi re")
    assert res_frag2["clause_type"] == "verbless_fragment"

    # Simple clause
    res_simple = profiler.analyze_turn_syntax("mera hotel oa baki hai abhi")
    assert res_simple["clause_type"] == "simple_clause"

    # Compound clause (coordinating conjunction 'aur')
    res_comp = profiler.analyze_turn_syntax("mai kal aunga aur code push kar dunga")
    assert res_comp["clause_type"] == "compound_coordinate"

    # Complex clause (subordinating conjunction 'kyuki')
    res_sub = profiler.analyze_turn_syntax("kyuki dev pe bhi same error aa raha hai")
    assert res_sub["clause_type"] == "complex_subordinate"


def test_analyze_turn_speech_acts(profiler):
    # Imperative (command)
    res_imp = profiler.analyze_turn_syntax("check kar backend api")
    assert res_imp["is_imperative"] is True

    res_imp2 = profiler.analyze_turn_syntax("padh ya soja chupchap")
    assert res_imp2["is_imperative"] is True

    # Negative imperative
    res_mat = profiler.analyze_turn_syntax("mat bol usko kuch")
    assert res_mat["is_imperative"] is True
    assert res_mat["has_negation"] is True

    # Inquisitive with question mark
    res_q1 = profiler.analyze_turn_syntax("kya hua?")
    assert res_q1["is_inquisitive"] is True

    # Inquisitive without question mark (Wh-word)
    res_q2 = profiler.analyze_turn_syntax("kab tak aayega result")
    assert res_q2["is_inquisitive"] is True


def test_analyze_turn_negation_and_conditionals(profiler):
    # Negation
    res_neg = profiler.analyze_turn_syntax("nahi re chodd kuch nahi hua")
    assert res_neg["has_negation"] is True
    assert res_neg["negation_count"] >= 2

    # Conditional
    res_cond = profiler.analyze_turn_syntax("agar interview clear ho gaya toh parties karenge")
    assert res_cond["has_conditional"] is True


def test_analyze_turn_pronoun_orientation(profiler):
    # 1st person
    res_1st = profiler.analyze_turn_syntax("mene bola tha mere laptop pe chalu hai")
    assert res_1st["has_1st_person"] is True
    assert res_1st["has_2nd_person"] is False

    # 2nd person
    res_2nd = profiler.analyze_turn_syntax("tu bata tera kaisa gaya test")
    assert res_2nd["has_2nd_person"] is True


def test_fit_corpus_syntactic_profile(profiler):
    sample_pairs = [
        {"pair_id": "p1", "target_text": "check kar backend endpoint"},
        {"pair_id": "p2", "target_text": "kya bola usne fir?"},
        {"pair_id": "p3", "target_text": "nahi re kuch nahi"},
        {"pair_id": "p4", "target_text": "agar issue aaye toh bata dena"},
        {"pair_id": "p5", "target_text": "ha"},
    ]

    prof = profiler.fit(sample_pairs, name="test_corpus")

    assert isinstance(prof, SyntacticProfile)
    assert prof.total_turns == 5

    # Clause structure ratios
    clauses = prof.clause_structure
    assert 0.0 <= clauses["verbless_fragment_ratio"] <= 1.0
    assert 0.0 <= clauses["simple_clause_ratio"] <= 1.0

    # Speech acts
    acts = prof.speech_acts
    assert acts["imperative_directive_ratio"] > 0
    assert acts["inquisitive_question_ratio"] > 0

    # Negation and conditionals
    neg = prof.negation_and_conditionals
    assert neg["negation_turns_ratio"] > 0
    assert neg["conditional_turns_ratio"] > 0


def test_analyze_by_style(profiler):
    sample_pairs = [
        {"pair_id": "p1", "target_text": "check kar endpoint"},
        {"pair_id": "p2", "target_text": "run npm start"},
        {"pair_id": "p3", "target_text": "nahi yaar kuch nahi"},
        {"pair_id": "p4", "target_text": "nahi chal raha"},
    ]
    style_labels = np.array([4, 4, 0, 0], dtype=int)
    style_names = {4: "Technical_Collab", 0: "Denial_Friction"}

    profiles = profiler.analyze_by_style(sample_pairs, style_labels, style_names)

    assert "Technical_Collab" in profiles
    assert "Denial_Friction" in profiles

    tech = profiles["Technical_Collab"]
    denial = profiles["Denial_Friction"]

    # Tech should have higher imperative ratio than Denial
    assert tech.speech_acts["imperative_directive_ratio"] > denial.speech_acts["imperative_directive_ratio"]
    # Denial should have higher negation ratio than Tech
    assert denial.negation_and_conditionals["negation_turns_ratio"] > tech.negation_and_conditionals["negation_turns_ratio"]


def test_empty_input_raises_error(profiler):
    with pytest.raises(ValueError, match="cannot be empty"):
        profiler.fit([])
