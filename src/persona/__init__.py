"""
Persona modeling package for global and situational styles,
LLM inference interfaces, and persona packaging.
"""

from src.persona.evidence import EvidenceAggregator, EvidenceItem, EvidenceRegistry
from src.persona.inference import (
    EpistemicStatus,
    PersonaInferenceItem,
    StructuredInferenceEngine,
    StructuredInferenceResult,
)
from src.persona.style import GlobalStyleProfile, SituationalStyleProfile, StyleProfileBuilder
from src.persona.style_examples import (
    StyleExample,
    StyleExampleBuilder,
    StyleExampleConfig,
    StyleExampleResult,
    extract_style_tags,
)

__all__ = [
    "EpistemicStatus",
    "EvidenceAggregator",
    "EvidenceItem",
    "EvidenceRegistry",
    "GlobalStyleProfile",
    "PersonaInferenceItem",
    "SituationalStyleProfile",
    "StructuredInferenceEngine",
    "StructuredInferenceResult",
    "StyleExample",
    "StyleExampleBuilder",
    "StyleExampleConfig",
    "StyleExampleResult",
    "StyleProfileBuilder",
    "extract_style_tags",
]
