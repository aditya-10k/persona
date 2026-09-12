"""
Stages T022 & T023: Global Style Profile & Situational Style Profiles.
Synthesizes surface linguistics, syntax, code-switching, discourse functions,
and behavioral traits into cohesive, production-ready persona profiles
for unconstrained global generation and situation-conditioned modulation.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.nlp.behavior import BehavioralExtractor, BehavioralProfile
from src.nlp.code_switching import CodeSwitchingAnalyzer, LanguageProfile
from src.nlp.discourse import DiscourseAct, DiscourseAnalyzer, DiscourseProfile
from src.nlp.linguistics import LinguisticProfile, LinguisticProfiler
from src.nlp.situations import SituationalCategory, SituationalClassifier, SituationalProfile
from src.nlp.syntax import SyntacticProfile, SyntacticProfiler

logger = logging.getLogger(__name__)


@dataclass
class GlobalStyleProfile:
    """Production-ready global persona style profile with explicit empirical constraints."""

    total_turns: int
    core_dimensions: Dict[str, float]
    surface_constraints: Dict[str, Any]
    syntactic_constraints: Dict[str, Any]
    language_constraints: Dict[str, Any]
    discourse_moves: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_turns": self.total_turns,
            "core_dimensions": self.core_dimensions,
            "surface_constraints": self.surface_constraints,
            "syntactic_constraints": self.syntactic_constraints,
            "language_constraints": self.language_constraints,
            "discourse_moves": self.discourse_moves,
            "metadata": self.metadata,
        }


@dataclass
class SituationalStyleProfile:
    """Situation-conditioned style profile with specific behavioral and linguistic modulations."""

    situation: str
    description: str
    total_turns: int
    situation_prevalence: float
    core_dimensions: Dict[str, float]
    surface_constraints: Dict[str, Any]
    language_constraints: Dict[str, Any]
    discourse_moves: Dict[str, Any]
    representative_exemplars: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "situation": self.situation,
            "description": self.description,
            "total_turns": self.total_turns,
            "situation_prevalence": round(self.situation_prevalence, 4),
            "core_dimensions": self.core_dimensions,
            "surface_constraints": self.surface_constraints,
            "language_constraints": self.language_constraints,
            "discourse_moves": self.discourse_moves,
            "representative_exemplars": self.representative_exemplars,
            "metadata": self.metadata,
        }


class StyleProfileBuilder:
    """
    Synthesizes multi-stage NLP extractions into unified Global and Situational Style Profiles.
    """

    def __init__(self):
        self.ling_profiler = LinguisticProfiler()
        self.syntax_profiler = SyntacticProfiler()
        self.cs_analyzer = CodeSwitchingAnalyzer()
        self.discourse_analyzer = DiscourseAnalyzer()
        self.situational_classifier = SituationalClassifier()
        self.behavioral_extractor = BehavioralExtractor()

    def build_global_style(
        self,
        pairs: Sequence[Dict[str, Any]],
        name: str = "global_persona",
    ) -> GlobalStyleProfile:
        """
        Extracts and synthesizes all empirical distributions into a unified GlobalStyleProfile.
        """
        n_turns = len(pairs)
        if n_turns == 0:
            raise ValueError("Pairs sequence cannot be empty for style profile generation.")

        ling_prof = self.ling_profiler.fit(pairs, name=name)
        syntax_prof = self.syntax_profiler.fit(pairs, name=name)
        cs_prof = self.cs_analyzer.fit(pairs, name=name)
        discourse_prof = self.discourse_analyzer.fit(pairs, name=name)
        beh_prof = self.behavioral_extractor.extract_profile(pairs, name=name)

        # Formality calculation:
        # Standard formal prose = punctuation (periods) + standard casing + low slang + low hinglish
        # High informality = zero punctuation (92.5%) + lowercase (43.5%) + high Hinglish
        punc_informal = ling_prof.punctuation_metrics["zero_terminal_punctuation_ratio"]
        case_informal = ling_prof.casing_distribution["all_lowercase_ratio"]
        formality_score = round(max(0.0, 1.0 - (punc_informal * 0.6 + case_informal * 0.4)), 4)

        core_dimensions = {
            "directness": beh_prof.dimensions["directness"].score,
            "formality": formality_score,
            "verbosity": beh_prof.dimensions["verbosity"].score,
            "humor": beh_prof.dimensions["humor_playfulness"].score,
            "sarcasm": round(beh_prof.dimensions["humor_playfulness"].score * 0.8, 4),
            "hedging": beh_prof.dimensions["hedging"].score,
            "emotional_expression": beh_prof.dimensions["emotional_expressiveness"].score,
            "question_initiative": beh_prof.dimensions["question_initiative"].score,
            "confidence": beh_prof.dimensions["confidence_assertion"].score,
            "code_switching_propensity": beh_prof.dimensions["code_switching_propensity"].score,
            "technical_depth": beh_prof.dimensions["technical_depth"].score,
        }

        surface_constraints = {
            "zero_terminal_punctuation_ratio": ling_prof.punctuation_metrics["zero_terminal_punctuation_ratio"],
            "all_lowercase_ratio": ling_prof.casing_distribution["all_lowercase_ratio"],
            "standard_title_case_ratio": ling_prof.casing_distribution["mixed_casing_ratio"],
            "all_caps_shouting_ratio": ling_prof.casing_distribution["all_uppercase_ratio"],
            "multi_bubble_burst_ratio": ling_prof.burstiness["multi_bubble_ratio"],
            "mean_bubbles_per_turn": ling_prof.burstiness["mean_messages_per_turn"],
            "emoji_turns_ratio": ling_prof.emoji_fingerprint["turns_with_emojis_ratio"],
            "emoji_burst_repetition_ratio": ling_prof.emoji_fingerprint["emoji_burst_ratio"],
            "top_emojis": ling_prof.emoji_fingerprint["top_emojis"][:5],
            "vocabulary_ttr": ling_prof.lexical_diversity["root_ttr_guiraud"],
        }

        syntactic_constraints = {
            "verbless_fragment_ratio": syntax_prof.clause_structure["verbless_fragment_ratio"],
            "simple_clause_ratio": syntax_prof.clause_structure["simple_clause_ratio"],
            "compound_coordinate_ratio": syntax_prof.clause_structure["compound_coordinate_ratio"],
            "complex_subordinate_ratio": syntax_prof.clause_structure["complex_subordinate_ratio"],
            "negation_turns_ratio": syntax_prof.negation_and_conditionals["negation_turns_ratio"],
            "self_to_other_pronoun_ratio": syntax_prof.pronoun_orientation["self_to_other_ratio"],
        }

        language_constraints = {
            "english_token_ratio": cs_prof.token_distribution["english_token_ratio"],
            "hindi_token_ratio": cs_prof.token_distribution["hindi_token_ratio"],
            "code_switched_turns_ratio": cs_prof.turn_classification["code_switched_ratio"],
            "pure_hindi_turns_ratio": cs_prof.turn_classification["pure_hindi_ratio"],
            "pure_english_turns_ratio": cs_prof.turn_classification["pure_english_ratio"],
            "mean_switches_per_turn": cs_prof.switching_dynamics["mean_switches_per_turn"],
            "top_borrowed_english_words": [b["word"] for b in cs_prof.borrowed_english_words[:10]],
        }

        discourse_moves = {
            "act_distribution": discourse_prof.act_distribution,
            "context_response_transitions": discourse_prof.context_response_transitions,
        }

        return GlobalStyleProfile(
            total_turns=n_turns,
            core_dimensions=core_dimensions,
            surface_constraints=surface_constraints,
            syntactic_constraints=syntactic_constraints,
            language_constraints=language_constraints,
            discourse_moves=discourse_moves,
            metadata={"name": name},
        )

    def build_situational_styles(
        self,
        pairs: Sequence[Dict[str, Any]],
        max_exemplars: int = 5,
    ) -> Dict[str, SituationalStyleProfile]:
        """
        Segments pairs by classified situation and generates tailored SituationalStyleProfiles.
        """
        n_turns = len(pairs)
        if n_turns == 0:
            raise ValueError("Pairs sequence cannot be empty for situational style profiling.")

        # Classify each turn into a situation
        situated_pairs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for p in pairs:
            text = p.get("target_text", "")
            ctx = p.get("context_text") or (p.get("context")[-1].get("text", "") if p.get("context") and isinstance(p.get("context")[-1], dict) else "")
            res = self.situational_classifier.classify_turn(text, context_text=ctx)
            situated_pairs[res.primary_situation].append(p)

        situation_descriptions = {
            SituationalCategory.TECHNICAL_COLLAB.value: "Technical collaboration, coding, API troubleshooting, deployment, and debugging.",
            SituationalCategory.CASUAL_BANTER.value: "Informal friendly banter, emotional venting, teasing, memes, and everyday chat.",
            SituationalCategory.CONFLICT_FRICTION.value: "Direct disagreement, skepticism, denial, pushing back on flawed premises.",
            SituationalCategory.CAREER_ACADEMIC.value: "College coursework, exam prep, placements, OA tests, interviews, and packages.",
            SituationalCategory.ACKNOWLEDGEMENT.value: "Rapid operational coordination, pings, status confirmations, and minimal check-ins.",
            SituationalCategory.ADVICE_PROBING.value: "Inquiring, probing questions, seeking troubleshooting guidance, and advice.",
        }

        situational_profiles: Dict[str, SituationalStyleProfile] = {}

        for sit_cat in SituationalCategory:
            sit_key = sit_cat.value
            group_pairs = situated_pairs.get(sit_key, [])
            count = len(group_pairs)
            prevalence = count / n_turns if n_turns > 0 else 0.0

            if count == 0:
                continue

            # Build behavioral & feature profiles for this specific situation
            beh_prof = self.behavioral_extractor.extract_profile(group_pairs, name=sit_key)
            ling_prof = self.ling_profiler.fit(group_pairs, name=sit_key)
            cs_prof = self.cs_analyzer.fit(group_pairs, name=sit_key)
            discourse_prof = self.discourse_analyzer.fit(group_pairs, name=sit_key)

            core_dims = {
                "directness": beh_prof.dimensions["directness"].score,
                "verbosity": beh_prof.dimensions["verbosity"].score,
                "humor": beh_prof.dimensions["humor_playfulness"].score,
                "hedging": beh_prof.dimensions["hedging"].score,
                "disagreement_style": beh_prof.dimensions["disagreement_style"].score,
                "confidence": beh_prof.dimensions["confidence_assertion"].score,
                "technical_depth": beh_prof.dimensions["technical_depth"].score,
                "code_switching_propensity": beh_prof.dimensions["code_switching_propensity"].score,
            }

            surface_const = {
                "zero_terminal_punctuation_ratio": ling_prof.punctuation_metrics["zero_terminal_punctuation_ratio"],
                "all_lowercase_ratio": ling_prof.casing_distribution["all_lowercase_ratio"],
                "emoji_turns_ratio": ling_prof.emoji_fingerprint["turns_with_emojis_ratio"],
                "mean_bubbles_per_turn": ling_prof.burstiness["mean_messages_per_turn"],
            }

            lang_const = {
                "english_token_ratio": cs_prof.token_distribution["english_token_ratio"],
                "hindi_token_ratio": cs_prof.token_distribution["hindi_token_ratio"],
                "code_switched_turns_ratio": cs_prof.turn_classification["code_switched_ratio"],
            }

            disc_moves = {
                "top_discourse_act": max(discourse_prof.act_distribution.items(), key=lambda x: x[1])[0],
                "act_distribution": discourse_prof.act_distribution,
            }

            # Mine representative exemplars
            exemplars = []
            seen_texts = set()
            for gp in group_pairs:
                txt = gp.get("target_text", "").strip()
                if txt and txt not in seen_texts and len(txt.split()) >= 2:
                    seen_texts.add(txt)
                    ctx = gp.get("context_text") or (gp.get("context")[-1].get("text", "") if gp.get("context") and isinstance(gp.get("context")[-1], dict) else "")
                    exemplars.append({
                        "pair_id": gp.get("pair_id", ""),
                        "context": ctx,
                        "response": txt,
                    })
                    if len(exemplars) >= max_exemplars:
                        break

            situational_profiles[sit_key] = SituationalStyleProfile(
                situation=sit_key,
                description=situation_descriptions.get(sit_key, sit_key),
                total_turns=count,
                situation_prevalence=prevalence,
                core_dimensions=core_dims,
                surface_constraints=surface_const,
                language_constraints=lang_const,
                discourse_moves=disc_moves,
                representative_exemplars=exemplars,
                metadata={"name": sit_key},
            )

        return situational_profiles
