"""
Unit tests for Stage T025: Evidence Aggregation.
Fast synthetic unit tests (< 1s) validating EvidenceRegistry creation,
evidence counts, confidence scores, and epistemic breakdown totals.
"""

from __future__ import annotations

import json
import pytest

from src.persona.evidence import EvidenceAggregator, EvidenceItem, EvidenceRegistry
from src.persona.inference import EpistemicStatus, StructuredInferenceEngine
from src.persona.style import GlobalStyleProfile, SituationalStyleProfile, StyleProfileBuilder


@pytest.fixture
def aggregator() -> EvidenceAggregator:
    return EvidenceAggregator()


def test_evidence_aggregation_and_validation(aggregator):
    sample_pairs = [
        {"pair_id": "p1", "target_text": "api endpoint curl deploy test", "context_text": "status check"},
        {"pair_id": "p2", "target_text": "ha pakka done", "context_text": "ho gaya?"},
        {"pair_id": "p3", "target_text": "nahi re bilkul galat", "context_text": "aisa hai"},
        {"pair_id": "p4", "target_text": "bc lol 😭😭", "context_text": "dekha ye"},
    ]

    builder = StyleProfileBuilder()
    global_style = builder.build_global_style(sample_pairs, name="test_global")
    situational_styles = builder.build_situational_styles(sample_pairs, max_exemplars=2)

    inference_engine = StructuredInferenceEngine()
    inference_result = inference_engine.infer_offline(global_style.to_dict(), {})

    registry = aggregator.aggregate(
        global_style=global_style,
        situational_styles=situational_styles,
        inference_result=inference_result,
        exemplar_pairs=sample_pairs,
    )

    assert isinstance(registry, EvidenceRegistry)
    assert registry.total_evidence_items > 5
    assert registry.total_evaluated_turns == 4

    # Validate individual items
    for key, item in registry.items.items():
        assert isinstance(item, EvidenceItem)
        assert 0.0 <= item.score <= 1.0
        assert 0.0 <= item.confidence <= 1.0
        assert item.epistemic_status in {"observed", "inferred", "unknown"}
        assert len(item.rationale) > 0

    # Epistemic breakdown sum
    breakdown = registry.epistemic_breakdown
    assert breakdown["observed"] > 0
    assert breakdown["inferred"] > 0
    assert breakdown["unknown"] > 0
    assert sum(breakdown.values()) == registry.total_evidence_items

    # JSON serialization
    as_dict = registry.to_dict()
    assert json.loads(json.dumps(as_dict)) == as_dict
