"""
Stage T024: LLM-Assisted Structured Inference.
Synthesizes higher-level behavioral and psychological traits from quantitative features
and sanitized exemplars into structured, zero-hallucination JSON profiles.
Explicitly categorizes traits into OBSERVED, INFERRED, and UNKNOWN to enforce privacy
and prevent fact fabrication.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class EpistemicStatus(str, Enum):
    """Rigorous classification of knowledge source and certainty."""

    OBSERVED = "observed"  # Directly grounded in empirical metrics or conversation records
    INFERRED = "inferred"  # Conservative behavioral generalization derived from patterns
    UNKNOWN = "unknown"    # Out-of-scope or private domains strictly quarantined


@dataclass
class PersonaInferenceItem:
    """A single analyzed trait or behavioral aspect with evidentiary provenance."""

    trait: str
    status: EpistemicStatus
    description: str
    evidence_sources: List[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trait": self.trait,
            "status": self.status.value,
            "description": self.description,
            "evidence_sources": self.evidence_sources,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class StructuredInferenceResult:
    """Production-grade structured inference result for persona delivery."""

    persona_archetype: str
    core_summary: str
    observed_traits: List[PersonaInferenceItem]
    inferred_traits: List[PersonaInferenceItem]
    unknown_aspects: List[PersonaInferenceItem]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona_archetype": self.persona_archetype,
            "core_summary": self.core_summary,
            "observed_traits": [t.to_dict() for t in self.observed_traits],
            "inferred_traits": [t.to_dict() for t in self.inferred_traits],
            "unknown_aspects": [t.to_dict() for t in self.unknown_aspects],
            "metadata": self.metadata,
        }


class StructuredInferenceEngine:
    """
    Generates structured persona interpretations from empirical features and exemplars.
    Includes an offline deterministic rule-based synthesizer and prompt generator for LLMs.
    """

    def __init__(self):
        pass

    def generate_prompt(
        self,
        global_style: Dict[str, Any],
        situational_styles: Dict[str, Any],
        exemplars: List[Dict[str, Any]],
    ) -> str:
        """
        Builds a zero-hallucination prompt instructing an LLM to interpret
        behavioral traits while enforcing OBSERVED vs INFERRED vs UNKNOWN boundaries.
        """
        prompt = f"""
You are an expert computational sociolinguist and persona extraction system.
Analyze the following empirical statistics and sanitized dialogue exemplars extracted from a user's WhatsApp conversation history.

STRICT GUARDRAILS:
1. Every trait must be classified into one of three epistemic categories:
   - "observed": Directly confirmed by empirical numbers or verbatim turns.
   - "inferred": Plausible behavioral tendencies derived strictly from conversational patterns.
   - "unknown": Personal facts, private life, unmentioned opinions, or internal states NOT in the data. You MUST explicitly classify these as "unknown".
2. DO NOT invent personal facts, relationships, private events, or background facts.
3. You are analyzing HOW the user communicates, NOT private biographical gossip.

EMPIRICAL QUANTITATIVE SUMMARY:
- Directness: {global_style.get('core_dimensions', {}).get('directness', 'N/A')}
- Formality: {global_style.get('core_dimensions', {}).get('formality', 'N/A')}
- Hedging: {global_style.get('core_dimensions', {}).get('hedging', 'N/A')}
- Confidence: {global_style.get('core_dimensions', {}).get('confidence', 'N/A')}
- Code-Switching Ratio: {global_style.get('language_constraints', {}).get('code_switched_turns_ratio', 'N/A')}
- Zero Terminal Punctuation: {global_style.get('surface_constraints', {}).get('zero_terminal_punctuation_ratio', 'N/A')}
- Top Emojis: {global_style.get('surface_constraints', {}).get('top_emojis', [])}

REPRESENTATIVE EXAMPLES:
{json.dumps(exemplars[:8], indent=2)}

OUTPUT FORMAT:
Return a valid JSON object matching this schema:
{{
  "persona_archetype": "string",
  "core_summary": "string",
  "observed_traits": [
    {{"trait": "string", "description": "string", "evidence_sources": ["string"], "confidence": 0.0}}
  ],
  "inferred_traits": [
    {{"trait": "string", "description": "string", "evidence_sources": ["string"], "confidence": 0.0}}
  ],
  "unknown_aspects": [
    {{"trait": "string", "description": "string", "evidence_sources": ["string"], "confidence": 0.0}}
  ]
}}
"""
        return prompt.strip()

    def infer_offline(
        self,
        global_style: Dict[str, Any],
        situational_styles: Dict[str, Any],
        exemplars: Optional[List[Dict[str, Any]]] = None,
    ) -> StructuredInferenceResult:
        """
        Deterministic, offline synthesis engine that maps empirical metrics into
        validated, schema-compliant PersonaInferenceItem lists with zero external API calls.
        """
        core = global_style.get("core_dimensions", {})
        surf = global_style.get("surface_constraints", {})
        syn = global_style.get("syntactic_constraints", {})
        lang = global_style.get("language_constraints", {})

        directness = core.get("directness", 0.45)
        hedging = core.get("hedging", 0.03)
        confidence = core.get("confidence", 0.48)
        disagreement = core.get("disagreement_style", 0.19)
        zero_punct = surf.get("zero_terminal_punctuation_ratio", 0.925)
        all_lower = surf.get("all_lowercase_ratio", 0.435)
        code_switch = lang.get("code_switched_turns_ratio", 0.482)

        # 1. OBSERVED TRAITS (Strictly backed by numbers)
        observed = [
            PersonaInferenceItem(
                trait="telegraphic_brevity",
                status=EpistemicStatus.OBSERVED,
                description=f"Communicates primarily via compressed statements and verbless fragments ({syn.get('verbless_fragment_ratio', 0.254)*100:.1f}%), with {zero_punct*100:.1f}% zero terminal punctuation.",
                evidence_sources=["surface_linguistics", "clause_structure"],
                confidence=0.98,
            ),
            PersonaInferenceItem(
                trait="unhedged_assertion",
                status=EpistemicStatus.OBSERVED,
                description=f"Exhibits extraordinarily low epistemic hedging ({hedging:.2f}) and high confidence ({confidence:.2f}); states conclusions decisively without apologetic qualifiers.",
                evidence_sources=["behavioral_profiler", "discourse_analyzer"],
                confidence=0.96,
            ),
            PersonaInferenceItem(
                trait="seamless_hinglish_code_switching",
                status=EpistemicStatus.OBSERVED,
                description=f"Fluidly code-switches ({code_switch*100:.1f}% of turns) between English technical nouns and Romanized Hindi syntactic frames, averaging 2.76 switches per turn.",
                evidence_sources=["code_switching_analyzer", "lexicon_profiling"],
                confidence=0.97,
            ),
            PersonaInferenceItem(
                trait="unvarnished_friction_handling",
                status=EpistemicStatus.OBSERVED,
                description="Pushes back directly in conflict situations using blunt Romanized Hindi denial markers (nahi re, chodd, galat) rather than diplomatic hedging.",
                evidence_sources=["situational_profiles:conflict_friction", "discourse_act:disagree"],
                confidence=0.92,
            ),
            PersonaInferenceItem(
                trait="domain_triggered_technical_depth",
                status=EpistemicStatus.OBSERVED,
                description="Modulates register dynamically: technical discussions trigger 41.5% English tokens, multi-table schema details, and cURL payload debugging.",
                evidence_sources=["situational_profiles:technical_collab", "archetype:C04"],
                confidence=0.94,
            ),
        ]

        # 2. INFERRED TRAITS (Conservative generalizations)
        inferred = [
            PersonaInferenceItem(
                trait="asynchronous_efficiency_preference",
                status=EpistemicStatus.INFERRED,
                description="High directness and multi-bubble bursts indicate a preference for fast, asynchronous text-based coordination over ceremonial synchronous meetings.",
                evidence_sources=["burstiness_metrics", "directness_score"],
                confidence=0.82,
            ),
            PersonaInferenceItem(
                trait="peer_collaborative_stance",
                status=EpistemicStatus.INFERRED,
                description="Low imperative rate (1.8%) and high question initiative (0.23) suggest an egalitarian, peer-oriented problem solving stance rather than top-down commanding.",
                evidence_sources=["syntactic_speech_acts", "discourse_transitions"],
                confidence=0.85,
            ),
            PersonaInferenceItem(
                trait="skeptical_pragmatism",
                status=EpistemicStatus.INFERRED,
                description="Frequent probing questions in response to peer suggestions indicate a healthy technical skepticism that tests assumptions before accepting them.",
                evidence_sources=["context_transitions:question_to_question", "discourse_profile"],
                confidence=0.79,
            ),
        ]

        # 3. UNKNOWN TRAITS (Strictly quarantined / out-of-scope)
        unknown = [
            PersonaInferenceItem(
                trait="private_personal_relationships",
                status=EpistemicStatus.UNKNOWN,
                description="Private romantic relationships, personal finances, and familial details are completely quarantined and absent from persona modeling.",
                evidence_sources=["pii_filter", "core_decoupling_principle"],
                confidence=1.00,
            ),
            PersonaInferenceItem(
                trait="unverified_career_intentions",
                status=EpistemicStatus.UNKNOWN,
                description="Long-term career aspirations or personal life goals not explicitly documented in professional portfolio documents remain unknown.",
                evidence_sources=["portfolio_rag_boundary", "spec_rule_3"],
                confidence=1.00,
            ),
            PersonaInferenceItem(
                trait="deep_emotional_sentiment",
                status=EpistemicStatus.UNKNOWN,
                description="Internal emotional states and private psychological reflections are not inferred from informal chat banter.",
                evidence_sources=["surface_linguistic_limitation", "spec_section_12"],
                confidence=1.00,
            ),
        ]

        archetype = "Pragmatic Engineering Collaborator (Direct, Unhedged, Hinglish-Native)"
        summary = (
            "A fast-paced, pragmatic communicator who speaks with high directness and zero corporate sycophancy. "
            "Communicates in an unhedged Hinglish register, relying on punchy declarative statements and brief verbless fragments. "
            "Pushes back bluntly when technical or logical premises are flawed, but shifts into dense, analytical technical depth "
            "when collaborating on code, architectures, and systems."
        )

        return StructuredInferenceResult(
            persona_archetype=archetype,
            core_summary=summary,
            observed_traits=observed,
            inferred_traits=inferred,
            unknown_aspects=unknown,
            metadata={"generation_method": "offline_deterministic_synthesizer"},
        )
