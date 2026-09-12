"""
Unit tests for Stage T030–T034: Production Persona Package Compiler.
Runs in < 1s on synthetic fixtures without loading transformer weights.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.persona.package import PersonaPackageCompiler


@pytest.fixture
def compiler() -> PersonaPackageCompiler:
    return PersonaPackageCompiler(version="1.0.0")


@pytest.fixture
def sample_inputs() -> dict:
    return {
        "global_style": {
            "archetype_distribution": {"Reactive_Slang": 0.3},
            "dominant_archetype": "Reactive_Slang",
            "surface_markers": {"lowercase_ratio": 0.435, "zero_terminal_punct_ratio": 0.925, "multi_bubble_ratio": 0.444, "emoji_ratio": 0.066},
        },
        "situational_styles": {
            "situations": {
                "technical_collab": {"dominant_archetype": "Technical_Collab"},
                "casual_banter": {"dominant_archetype": "Reactive_Slang"},
            }
        },
        "linguistics": {
            "total_turns": 2750,
            "terminal_punctuation_distribution": {"none": 0.925, "question": 0.232, "exclamation": 0.038, "period": 0.021},
            "casing_distribution": {"lowercase": 0.435, "mixed": 0.548, "uppercase": 0.017},
            "mean_words_per_turn": 8.4,
            "multi_bubble_ratio": 0.444,
            "hapax_legomena_ratio": 0.506,
            "total_vocabulary_size": 4320,
        },
        "syntax": {
            "clause_distribution": {"verbless_fragment": 0.254, "simple_clause": 0.574, "compound_clause": 0.136, "complex_clause": 0.036},
            "speech_act_distribution": {"declarative": 0.702, "interrogative": 0.232, "imperative": 0.066},
            "negation_ratio": 0.216,
            "pronoun_distribution": {"self": 0.727, "other": 0.273},
            "self_to_other_ratio": 2.66,
        },
        "language": {
            "overall_code_switched_ratio": 0.482,
            "overall_token_distribution": {"hi_token_ratio": 0.728, "en_token_ratio": 0.272},
        },
        "behavior": {
            "dimensions": {
                "directness": {"score": 0.45, "classification": "moderate", "confidence": 0.85, "evidence_count": 2750, "description": "test"},
                "hedging": {"score": 0.03, "classification": "very_low", "confidence": 0.95, "evidence_count": 2750, "description": "test"},
            }
        },
        "discourse": {
            "dominant_acts": ["opinion_statement", "reactive_comment"],
            "top_transitions": {"opinion_statement": {"reactive_comment": 0.4}},
        },
        "inference": {
            "epistemic_breakdown": {
                "observed": [{"claim": "low hedging"}],
                "inferred": [{"claim": "active hours"}],
                "unknown": [{"claim": "passwords"}],
            }
        },
    }


def test_build_style_profile(compiler, sample_inputs):
    doc = compiler.build_style_profile(
        sample_inputs["global_style"],
        sample_inputs["situational_styles"],
        sample_inputs["language"],
    )
    assert doc["version"] == "1.0.0"
    assert doc["profile_type"] == "style_profile"
    assert "global_style" in doc
    assert doc["global_style"]["casing_rule"]["primary_mode"] == "lowercase_dominant"
    assert doc["global_style"]["punctuation_rule"]["terminal_punctuation_drop_rate"] == 0.925
    assert "technical_collab" in doc["situational_styles"]


def test_build_linguistic_stats(compiler, sample_inputs):
    doc = compiler.build_linguistic_stats(
        sample_inputs["linguistics"],
        sample_inputs["syntax"],
        sample_inputs["language"],
    )
    assert doc["profile_type"] == "linguistic_stats"
    assert doc["surface_statistics"]["terminal_punctuation"]["zero_terminal_punctuation"] == 0.925
    assert doc["syntactic_statistics"]["clause_structure"]["verbless_fragments"] == 0.254
    assert doc["language_mixing_statistics"]["matrix_language"] == "Hindi (Hinglish)"


def test_build_behavior_profile(compiler, sample_inputs):
    doc = compiler.build_behavior_profile(
        sample_inputs["behavior"],
        sample_inputs["discourse"],
        sample_inputs["inference"],
    )
    assert doc["profile_type"] == "behavior_profile"
    assert "directness" in doc["quantified_dimensions"]
    assert doc["quantified_dimensions"]["hedging"]["score"] == 0.03
    assert len(doc["epistemic_categories"]["observed"]) > 0
    assert len(doc["communication_boundaries"]) > 0


def test_build_vocabulary(compiler):
    doc = compiler.build_vocabulary()
    assert doc["profile_type"] == "vocabulary_inventory"
    assert len(doc["discourse_connectives"]) > 0
    assert len(doc["confirmation_and_acknowledgement"]) > 0
    assert len(doc["denial_and_friction"]) > 0
    assert len(doc["slang_and_expressive_markers"]) > 0
    assert len(doc["technical_terms"]) > 0
    assert len(doc["characteristic_collocations"]) > 0
    assert len(doc["top_emojis"]) > 0


def test_build_system_prompt(compiler, sample_inputs):
    style = compiler.build_style_profile(sample_inputs["global_style"], sample_inputs["situational_styles"], sample_inputs["language"])
    beh = compiler.build_behavior_profile(sample_inputs["behavior"], sample_inputs["discourse"], sample_inputs["inference"])
    vocab = compiler.build_vocabulary()

    prompt = compiler.build_system_prompt(style, beh, vocab)
    assert "# SYSTEM PROMPT: AUTHENTIC COMMUNICATION PERSONA" in prompt
    assert "NO TRAILING PERIODS" in prompt
    assert "CASING PREFERENCE" in prompt
    assert "NO ROBOTIC AI PLEASANTRIES" in prompt
    assert "LOW HEDGING" in prompt
    assert "{{FEW_SHOT_EXEMPLARS}}" in prompt


def test_build_persona_report(compiler, sample_inputs):
    style = compiler.build_style_profile(sample_inputs["global_style"], sample_inputs["situational_styles"], sample_inputs["language"])
    ling = compiler.build_linguistic_stats(sample_inputs["linguistics"], sample_inputs["syntax"], sample_inputs["language"])
    beh = compiler.build_behavior_profile(sample_inputs["behavior"], sample_inputs["discourse"], sample_inputs["inference"])
    vocab = compiler.build_vocabulary()

    report = compiler.build_persona_report(style, ling, beh, vocab)
    assert "# Forensic Communication Persona Report" in report
    assert "Executive Summary" in report
    assert "Linguistic Fingerprint" in report
    assert "Behavioral & Discourse Architecture" in report
    assert "Epistemic Guardrails" in report


def test_compile_end_to_end(compiler, sample_inputs, tmp_path):
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    (data_dir / "style").mkdir()
    (data_dir / "linguistics").mkdir()
    (data_dir / "syntax").mkdir()
    (data_dir / "language").mkdir()
    (data_dir / "behavior").mkdir()
    (data_dir / "discourse").mkdir()
    (data_dir / "inference").mkdir()

    with open(data_dir / "style" / "global_style_profile.json", "w", encoding="utf-8") as f:
        json.dump(sample_inputs["global_style"], f)
    with open(data_dir / "style" / "situational_style_profiles.json", "w", encoding="utf-8") as f:
        json.dump(sample_inputs["situational_styles"], f)
    with open(data_dir / "linguistics" / "global_linguistic_profile.json", "w", encoding="utf-8") as f:
        json.dump(sample_inputs["linguistics"], f)
    with open(data_dir / "syntax" / "global_syntactic_profile.json", "w", encoding="utf-8") as f:
        json.dump(sample_inputs["syntax"], f)
    with open(data_dir / "language" / "global_language_profile.json", "w", encoding="utf-8") as f:
        json.dump(sample_inputs["language"], f)
    with open(data_dir / "behavior" / "global_behavioral_profile.json", "w", encoding="utf-8") as f:
        json.dump(sample_inputs["behavior"], f)
    with open(data_dir / "discourse" / "global_discourse_profile.json", "w", encoding="utf-8") as f:
        json.dump(sample_inputs["discourse"], f)
    with open(data_dir / "inference" / "structured_persona_inference.json", "w", encoding="utf-8") as f:
        json.dump(sample_inputs["inference"], f)

    out_dir = tmp_path / "persona_package"
    artifacts = compiler.compile(data_dir, out_dir)

    assert len(artifacts) == 7
    for name, path in artifacts.items():
        assert path.exists(), f"Missing artifact: {name}"
        assert path.stat().st_size > 0, f"Empty artifact: {name}"

    # Verify package_metadata
    with open(artifacts["package_metadata"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["version"] == "1.0.0"
    assert len(meta["artifact_checksums_sha256"]) == 6
