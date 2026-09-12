"""
Unit tests for Stage T024: LLM-Assisted Structured Inference.
Fast synthetic unit tests (< 1s) validating zero-hallucination prompt generation,
epistemic status partitioning (OBSERVED, INFERRED, UNKNOWN), and offline inference.
"""

from __future__ import annotations

import json
import pytest

from src.persona.inference import (
    EpistemicStatus,
    PersonaInferenceItem,
    StructuredInferenceEngine,
    StructuredInferenceResult,
)


@pytest.fixture
def engine() -> StructuredInferenceEngine:
    return StructuredInferenceEngine()


def test_generate_prompt_structure(engine):
    global_style = {
        "core_dimensions": {"directness": 0.45, "formality": 0.27, "hedging": 0.03, "confidence": 0.48},
        "surface_constraints": {"zero_terminal_punctuation_ratio": 0.925, "top_emojis": ["😭", "😂"]},
        "language_constraints": {"code_switched_turns_ratio": 0.482},
    }
    exemplars = [{"context": "hi", "response": "done"}]

    prompt = engine.generate_prompt(global_style, {}, exemplars)

    assert "STRICT GUARDRAILS" in prompt
    assert "observed" in prompt
    assert "inferred" in prompt
    assert "unknown" in prompt
    assert "0.45" in prompt
    assert "0.925" in prompt


def test_infer_offline_epistemic_categories(engine):
    global_style = {
        "core_dimensions": {"directness": 0.45, "hedging": 0.03, "confidence": 0.48},
        "surface_constraints": {"zero_terminal_punctuation_ratio": 0.925},
        "syntactic_constraints": {"verbless_fragment_ratio": 0.254},
        "language_constraints": {"code_switched_turns_ratio": 0.482},
    }

    result = engine.infer_offline(global_style, {})

    assert isinstance(result, StructuredInferenceResult)
    assert len(result.persona_archetype) > 0
    assert len(result.core_summary) > 0

    # Check observed traits
    assert len(result.observed_traits) >= 3
    for item in result.observed_traits:
        assert item.status == EpistemicStatus.OBSERVED
        assert 0.0 <= item.confidence <= 1.0

    # Check inferred traits
    assert len(result.inferred_traits) >= 2
    for item in result.inferred_traits:
        assert item.status == EpistemicStatus.INFERRED
        assert 0.0 <= item.confidence <= 1.0

    # Check unknown aspects
    assert len(result.unknown_aspects) >= 2
    for item in result.unknown_aspects:
        assert item.status == EpistemicStatus.UNKNOWN
        assert item.confidence == 1.00

    # Check JSON serialization
    as_dict = result.to_dict()
    assert json.loads(json.dumps(as_dict)) == as_dict
