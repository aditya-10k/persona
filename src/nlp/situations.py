"""
Stage T020: Situational Classification.
Classifies conversational contexts and environments into distinct operational situations
(technical collaboration, casual banter, conflict/friction, career/academic, acknowledgement, advice/probing).
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


class SituationalCategory(str, Enum):
    """Core situational environments in conversational interactions."""

    TECHNICAL_COLLAB = "technical_collab"      # APIs, coding, deployment, infrastructure, debugging
    CASUAL_BANTER = "casual_banter"            # Informal chat, jokes, everyday banter, hanging out
    CONFLICT_FRICTION = "conflict_friction"    # Disagreements, skepticism, pushback, denial
    CAREER_ACADEMIC = "career_academic"        # Placement, college, exams, interviews, job offers
    ACKNOWLEDGEMENT = "acknowledgement"        # Coordination, pings, brief status confirmations
    ADVICE_PROBING = "advice_probing"          # Inquiring, seeking recommendations, troubleshooting guidance


# Lexical patterns for situations
RE_TECH = re.compile(
    r"\b(api|apis|curl|endpoint|payload|json|xml|token|server|backend|frontend|code|repo|branch|commit|pr|merge|push|pull|bug|issue|error|exception|debug|deploy|build|docker|aws|sql|database|table|query|schema|http|https|postman|test|angular|react|node|python|java|ts|js|script|cdta|ach|air|flight|carrier|hotel|inr)\b",
    re.IGNORECASE,
)

RE_CAREER = re.compile(
    r"\b(placement|placements|company|interview|interviews|round|rounds|package|lpa|ctc|resume|cv|job|offer|intern|internship|clg|college|sem|semester|exam|exams|test|paper|marks|cgpa|oa|online assessment)\b",
    re.IGNORECASE,
)

RE_CONFLICT = re.compile(
    r"\b(nahi|nhi|galat|wrong|chodd|chod na|kuch nahi|kuch nhi|aisa nahi|aisa nhi|bakchodi|chutiya|bkl|bc|fati|disagree|nope|nah|false)\b",
    re.IGNORECASE,
)

RE_ACK = re.compile(
    r"^(ha|haa|haan|ok|okay|k|kk|done|cool|nice|sahi|thik|theek|hmm|han|shi|yep|yes)$",
    re.IGNORECASE,
)

RE_ADVICE_PROBING = re.compile(
    r"(\?|\b(kya|kyu|kyun|kaise|kese|kaha|kahan|kidhar|kitna|kitne|kab|who|what|why|how|when|where|which|try kar|try karo|check kar|batao|btao|bata na|suggest|advice)\b)",
    re.IGNORECASE,
)

RE_EMOJI_BANTER = re.compile(r"[\U0001f602\U0001f923\U0001f62d\U0001f480\U0001f643\U0001f921\U0001f4a9]")


@dataclass
class SituationResult:
    """Classification result for a single interaction turn."""

    primary_situation: str
    confidence: float
    scores: Dict[str, float]
    cues_detected: Dict[str, List[str]]


@dataclass
class SituationalProfile:
    """Corpus-level situational distribution and transition dynamics."""

    total_turns: int
    situation_distribution: Dict[str, float]
    situation_counts: Dict[str, int]
    top_cues_by_situation: Dict[str, List[Dict[str, Any]]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_turns": self.total_turns,
            "situation_distribution": self.situation_distribution,
            "situation_counts": self.situation_counts,
            "top_cues_by_situation": self.top_cues_by_situation,
            "metadata": self.metadata,
        }


class SituationalClassifier:
    """
    Classifies conversational turns and contexts into situational categories
    based on multi-level lexical cues, length, and dialogue structure.
    """

    def __init__(self):
        pass

    def classify_turn(
        self,
        text: str,
        context_text: Optional[str] = None,
    ) -> SituationResult:
        """
        Classifies a single interaction turn into its primary situation.
        Uses combined signals from the target text and preceding context.
        """
        raw_text = text.strip()
        combined_text = f"{context_text or ''} {raw_text}".strip().lower()

        cues: Dict[str, List[str]] = defaultdict(list)
        scores: Dict[str, float] = {sit.value: 0.0 for sit in SituationalCategory}

        # 1. Acknowledgement check (high priority for very brief target responses)
        words = raw_text.split()
        if len(words) <= 3 and RE_ACK.match(raw_text):
            scores[SituationalCategory.ACKNOWLEDGEMENT.value] += 3.0
            cues[SituationalCategory.ACKNOWLEDGEMENT.value].append(raw_text.lower())

        # 2. Technical Collab check
        tech_matches = RE_TECH.findall(combined_text)
        if tech_matches:
            weight = len(tech_matches) * 1.2
            scores[SituationalCategory.TECHNICAL_COLLAB.value] += weight
            cues[SituationalCategory.TECHNICAL_COLLAB.value].extend(tech_matches[:10])

        # 3. Career & Academic check
        career_matches = RE_CAREER.findall(combined_text)
        if career_matches:
            weight = len(career_matches) * 1.2
            scores[SituationalCategory.CAREER_ACADEMIC.value] += weight
            cues[SituationalCategory.CAREER_ACADEMIC.value].extend(career_matches[:10])

        # 4. Conflict & Friction check
        conflict_matches = RE_CONFLICT.findall(raw_text)
        if conflict_matches:
            weight = len(conflict_matches) * 1.0
            scores[SituationalCategory.CONFLICT_FRICTION.value] += weight
            cues[SituationalCategory.CONFLICT_FRICTION.value].extend(conflict_matches[:10])

        # 5. Advice & Probing check
        probing_matches = RE_ADVICE_PROBING.findall(combined_text)
        if probing_matches:
            matches = [m[0] if isinstance(m, tuple) else m for m in probing_matches]
            weight = len(matches) * 0.8
            scores[SituationalCategory.ADVICE_PROBING.value] += weight
            cues[SituationalCategory.ADVICE_PROBING.value].extend(matches[:10])

        # 6. Casual Banter check (emojis, slang, default)
        banter_emojis = RE_EMOJI_BANTER.findall(raw_text)
        if banter_emojis:
            scores[SituationalCategory.CASUAL_BANTER.value] += len(banter_emojis) * 0.8
            cues[SituationalCategory.CASUAL_BANTER.value].extend(banter_emojis[:10])

        # Baseline casual prior
        scores[SituationalCategory.CASUAL_BANTER.value] += 0.2

        # Normalize scores to pseudo-probabilities
        total_score = sum(scores.values())
        norm_scores = {k: round(v / total_score, 4) for k, v in scores.items()}

        # Pick primary situation
        primary = max(norm_scores.items(), key=lambda x: x[1])[0]
        confidence = norm_scores[primary]

        return SituationResult(
            primary_situation=primary,
            confidence=confidence,
            scores=norm_scores,
            cues_detected=dict(cues),
        )

    def fit(self, pairs: Sequence[Dict[str, Any]], name: str = "global") -> SituationalProfile:
        """
        Fits the SituationalClassifier across a corpus of pairs.
        """
        n_turns = len(pairs)
        if n_turns == 0:
            raise ValueError("Pairs sequence cannot be empty for situational classification.")

        situation_counts: Counter[str] = Counter()
        cue_counters: Dict[str, Counter[str]] = defaultdict(Counter)

        for p in pairs:
            target_text = p.get("target_text", "")
            context_text = None
            if p.get("context_text"):
                context_text = p["context_text"]
            elif p.get("context"):
                last_c = p["context"][-1]
                context_text = last_c.get("text", "") if isinstance(last_c, dict) else str(last_c)

            res = self.classify_turn(target_text, context_text=context_text)
            situation_counts[res.primary_situation] += 1

            for sit, c_list in res.cues_detected.items():
                for c in c_list:
                    cue_counters[sit][c.lower()] += 1

        sit_dist = {
            sit.value: round(situation_counts[sit.value] / n_turns, 4)
            for sit in SituationalCategory
        }

        top_cues: Dict[str, List[Dict[str, Any]]] = {}
        for sit in SituationalCategory:
            top_c = [
                {"cue": c, "count": cnt}
                for c, cnt in cue_counters[sit.value].most_common(10)
            ]
            top_cues[sit.value] = top_c

        return SituationalProfile(
            total_turns=n_turns,
            situation_distribution=sit_dist,
            situation_counts=dict(situation_counts),
            top_cues_by_situation=top_cues,
            metadata={"name": name},
        )

    def analyze_by_style(
        self,
        pairs: Sequence[Dict[str, Any]],
        style_labels: Sequence[int],
        style_names: Optional[Dict[int, str]] = None,
    ) -> Dict[str, SituationalProfile]:
        """
        Segments pairs by their communication archetype label and fits
        a SituationalProfile for each archetype.
        """
        if len(pairs) != len(style_labels):
            raise ValueError("Pairs and style_labels must have the exact same length.")

        style_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for pair, label in zip(pairs, style_labels):
            style_groups[int(label)].append(pair)

        profiles: Dict[str, SituationalProfile] = {}
        for label, group_pairs in sorted(style_groups.items()):
            name = (
                style_names.get(label, f"Style_{label}")
                if style_names
                else f"Style_{label}"
            )
            profiles[name] = self.fit(group_pairs, name=name)

        return profiles
