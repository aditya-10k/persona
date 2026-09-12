"""
Stage T019: Discourse Analysis.
Quantifies conversational discourse acts, interpersonal positioning, epistemic hedging,
speech act transitions, and communicative moves across Hinglish conversational turns.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DiscourseAct(str, Enum):
    """Core communicative and discourse functions in conversational interactions."""

    AGREE = "agree"              # Affirmation, confirmation, consensus
    DISAGREE = "disagree"        # Contradiction, denial, friction, skepticism
    QUESTION = "question"        # Interrogative probing, status checks
    EXPLAIN = "explain"          # Causal reasoning, justification, mechanism
    CLARIFY = "clarify"          # Elaboration, repair, rephrasing, specifying
    HEDGE = "hedge"              # Epistemic uncertainty, tentativeness
    ADVICE = "advice"            # Guidance, recommendation, directive suggestion
    HUMOR = "humor"              # Laughter, playful teasing, ironic banter
    EMPHASIS = "emphasis"        # Intensifiers, extreme polarity
    INFORMATIVE = "informative"  # Direct factual assertion, neutral information


# Patterns compiled for high performance
RE_HUMOR_EMOJIS = re.compile(r"[\U0001f602\U0001f923\U0001f62d\U0001f480\U0001f643\U0001f921\U0001f4a9\U0001f92e]")
RE_HUMOR_LEXICAL = re.compile(r"\b(lol|lmao|rofl|xd|lmfao|joke|mazak|mzaak|chutiyapa|bakchodi|pagal|bc|meme|memes)\b", re.IGNORECASE)

RE_AGREE = re.compile(
    r"\b(ha|haa|haan|hn|han|yep|yeah|yes|ok|okay|k|kk|cool|done|sahi|true|perfect|nice|fine|bilkul|agreed|agree|definitely|done bhai|pakka|exact|exactly|makes sense|good point|nice one|badhiya|badiya|shi)\b",
    re.IGNORECASE,
)

RE_DISAGREE = re.compile(
    r"\b(nahi|nhi|nope|nah|na|galat|aisa nahi|aisa nhi|chodd|chod na|kuch nahi|kuch nhi|mat kar|chutiya|no way|wrong|disagree|false|not true|dont|don't|nhina)\b",
    re.IGNORECASE,
)

RE_QUESTION = re.compile(
    r"(\?|\b(kya|kyu|kyun|kaise|kese|kaha|kahan|kidhar|kitna|kitne|kitni|kab|kon|kaun|who|what|why|how|when|where|which|whose|whom|batao|btao|bata na|hai kya|h kya)\b)",
    re.IGNORECASE,
)

RE_EXPLAIN = re.compile(
    r"\b(kyuki|kyunki|because|since|reason|reason ye hai|isliye|islie|cause|logic|isse hota|matlab ye tha|actually why|due to)\b",
    re.IGNORECASE,
)

RE_CLARIFY = re.compile(
    r"\b(matlab|i mean|specifically|specifically ye|actually|basically|in short|simple words|yaani|kehna ye|wait|correction|my bad|to be specific)\b",
    re.IGNORECASE,
)

RE_HEDGE = re.compile(
    r"\b(shayad|shyad|lagra|lagta|lag raha|lag rahi|lagte|maybe|probably|not sure|pata nahi|pata nhi|maloom nahi|i think|i guess|might be|could be|seems like|around|almost|kind of|sort of)\b",
    re.IGNORECASE,
)

RE_ADVICE = re.compile(
    r"\b(karo|karle|karna chahiye|chahiye|try kar|try karo|try karna|dekh le|dekh lo|check kar|check karo|mat kar|should|better to|recommend|suggest|idea ye hai|aise kar|aise karo)\b",
    re.IGNORECASE,
)

RE_EMPHASIS = re.compile(
    r"\b(literally|totally|definitely|absolutely|completely|bohot|bahut|zyada|jyada|sach me|sach bata|fr|fr fr|damn|huge|insane|crazy|extreme|bhai_bhai)\b",
    re.IGNORECASE,
)


@dataclass
class TurnDiscourseResult:
    """Discourse classification and communicative intent for a single turn."""

    primary_act: str
    act_tags: List[str]  # multi-label set of detected acts
    act_scores: Dict[str, float]  # normalized score / presence
    markers_detected: Dict[str, List[str]]
    context_act: Optional[str] = None


@dataclass
class DiscourseProfile:
    """Comprehensive discourse profile for a corpus or communication archetype."""

    total_turns: int
    act_distribution: Dict[str, float]
    act_counts: Dict[str, int]
    co_occurrence_matrix: Dict[str, Dict[str, int]]
    context_response_transitions: Dict[str, Dict[str, float]]
    top_markers_by_act: Dict[str, List[Dict[str, Any]]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_turns": self.total_turns,
            "act_distribution": self.act_distribution,
            "act_counts": self.act_counts,
            "co_occurrence_matrix": self.co_occurrence_matrix,
            "context_response_transitions": self.context_response_transitions,
            "top_markers_by_act": self.top_markers_by_act,
            "metadata": self.metadata,
        }


class DiscourseAnalyzer:
    """
    Extracts communicative discourse functions, interpersonal stance, and
    context-conditioned response dynamics from conversational turns.
    """

    def __init__(self):
        pass

    def analyze_turn(
        self,
        text: str,
        context_text: Optional[str] = None,
    ) -> TurnDiscourseResult:
        """
        Analyzes a single conversational turn for multi-label discourse acts
        and resolves a single dominant primary act.
        """
        raw_text = text.strip()
        if not raw_text:
            return TurnDiscourseResult(
                primary_act=DiscourseAct.INFORMATIVE.value,
                act_tags=[DiscourseAct.INFORMATIVE.value],
                act_scores={act.value: 0.0 for act in DiscourseAct},
                markers_detected={},
                context_act=None,
            )

        detected_markers: Dict[str, List[str]] = defaultdict(list)

        # 1. Check Humor
        humor_emojis = RE_HUMOR_EMOJIS.findall(raw_text)
        humor_words = RE_HUMOR_LEXICAL.findall(raw_text)
        if humor_emojis or humor_words:
            detected_markers[DiscourseAct.HUMOR.value].extend(humor_emojis + humor_words)

        # 2. Check Question
        q_matches = RE_QUESTION.findall(raw_text)
        if q_matches:
            # Flat match list
            matches = [m[0] if isinstance(m, tuple) else m for m in q_matches]
            detected_markers[DiscourseAct.QUESTION.value].extend(matches)

        # 3. Check Disagree
        disagree_matches = RE_DISAGREE.findall(raw_text)
        if disagree_matches:
            detected_markers[DiscourseAct.DISAGREE.value].extend(disagree_matches)

        # 4. Check Agree
        agree_matches = RE_AGREE.findall(raw_text)
        if agree_matches:
            detected_markers[DiscourseAct.AGREE.value].extend(agree_matches)

        # 5. Check Explain
        explain_matches = RE_EXPLAIN.findall(raw_text)
        if explain_matches:
            detected_markers[DiscourseAct.EXPLAIN.value].extend(explain_matches)

        # 6. Check Clarify
        clarify_matches = RE_CLARIFY.findall(raw_text)
        if clarify_matches:
            detected_markers[DiscourseAct.CLARIFY.value].extend(clarify_matches)

        # 7. Check Hedge
        hedge_matches = RE_HEDGE.findall(raw_text)
        if hedge_matches:
            detected_markers[DiscourseAct.HEDGE.value].extend(hedge_matches)

        # 8. Check Advice
        advice_matches = RE_ADVICE.findall(raw_text)
        if advice_matches:
            detected_markers[DiscourseAct.ADVICE.value].extend(advice_matches)

        # 9. Check Emphasis
        emphasis_matches = RE_EMPHASIS.findall(raw_text)
        if emphasis_matches:
            detected_markers[DiscourseAct.EMPHASIS.value].extend(emphasis_matches)

        # 10. Multi-label tags
        act_tags = list(detected_markers.keys())
        if not act_tags:
            act_tags = [DiscourseAct.INFORMATIVE.value]

        # Calculate presence scores
        act_scores = {act.value: 1.0 if act.value in act_tags else 0.0 for act in DiscourseAct}

        # Resolve single primary act with priority ordering:
        # Question > Disagree > Agree > Explain > Advice > Clarify > Hedge > Humor > Emphasis > Informative
        primary_act = DiscourseAct.INFORMATIVE.value
        words = raw_text.split()

        if "?" in raw_text or (DiscourseAct.QUESTION.value in act_tags and len(words) <= 7):
            primary_act = DiscourseAct.QUESTION.value
        elif DiscourseAct.DISAGREE.value in act_tags and (len(words) <= 6 or raw_text.lower().startswith(("nahi", "nhi", "no"))):
            primary_act = DiscourseAct.DISAGREE.value
        elif DiscourseAct.AGREE.value in act_tags and (len(words) <= 4 or raw_text.lower().startswith(("ha", "haa", "done", "ok", "cool"))):
            primary_act = DiscourseAct.AGREE.value
        elif DiscourseAct.EXPLAIN.value in act_tags:
            primary_act = DiscourseAct.EXPLAIN.value
        elif DiscourseAct.ADVICE.value in act_tags:
            primary_act = DiscourseAct.ADVICE.value
        elif DiscourseAct.CLARIFY.value in act_tags:
            primary_act = DiscourseAct.CLARIFY.value
        elif DiscourseAct.DISAGREE.value in act_tags:
            primary_act = DiscourseAct.DISAGREE.value
        elif DiscourseAct.AGREE.value in act_tags:
            primary_act = DiscourseAct.AGREE.value
        elif DiscourseAct.QUESTION.value in act_tags:
            primary_act = DiscourseAct.QUESTION.value
        elif DiscourseAct.HEDGE.value in act_tags:
            primary_act = DiscourseAct.HEDGE.value
        elif DiscourseAct.HUMOR.value in act_tags:
            primary_act = DiscourseAct.HUMOR.value
        elif DiscourseAct.EMPHASIS.value in act_tags:
            primary_act = DiscourseAct.EMPHASIS.value
        else:
            primary_act = DiscourseAct.INFORMATIVE.value

        # Classify context act if context_text is provided
        context_act = None
        if context_text is not None:
            if isinstance(context_text, dict):
                context_str = context_text.get("text", "")
            else:
                context_str = str(context_text)
            # Strip speaker prefix like "Speaker: " if present
            context_str = re.sub(r"^[^:\n]{1,30}:\s*", "", context_str).strip()
            if context_str:
                ctx_res = self.analyze_turn(context_str)
                context_act = ctx_res.primary_act

        return TurnDiscourseResult(
            primary_act=primary_act,
            act_tags=act_tags,
            act_scores=act_scores,
            markers_detected=dict(detected_markers),
            context_act=context_act,
        )

    def fit(self, pairs: Sequence[Dict[str, Any]], name: str = "global") -> DiscourseProfile:
        """
        Fits the DiscourseAnalyzer across a corpus of pairs to extract complete
        discourse distributions, co-occurrences, and context-response transitions.
        """
        n_turns = len(pairs)
        if n_turns == 0:
            raise ValueError("Pairs sequence cannot be empty for discourse profiling.")

        act_counts: Counter[str] = Counter()
        primary_counts: Counter[str] = Counter()
        co_occurrence: Dict[str, Dict[str, int]] = {
            a1.value: {a2.value: 0 for a2 in DiscourseAct} for a1 in DiscourseAct
        }
        transition_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        marker_counters: Dict[str, Counter[str]] = defaultdict(Counter)

        for p in pairs:
            target_text = p.get("target_text", "")
            # Extract last context turn if available
            context_text = None
            if p.get("context_text"):
                context_text = p["context_text"]
            elif p.get("context"):
                last_c = p["context"][-1]
                context_text = last_c.get("text", "") if isinstance(last_c, dict) else str(last_c)

            res = self.analyze_turn(target_text, context_text=context_text)

            primary_counts[res.primary_act] += 1
            for tag in res.act_tags:
                act_counts[tag] += 1

            # Update co-occurrence matrix
            for t1 in res.act_tags:
                for t2 in res.act_tags:
                    if t1 in co_occurrence and t2 in co_occurrence[t1]:
                        co_occurrence[t1][t2] += 1

            # Update context -> response transition
            if res.context_act:
                transition_counts[res.context_act][res.primary_act] += 1

            # Accumulate markers
            for act_name, m_list in res.markers_detected.items():
                for m in m_list:
                    marker_counters[act_name][m.lower()] += 1

        # Calculate normalized act distributions (based on primary act classification)
        act_dist = {
            act.value: round(primary_counts[act.value] / n_turns, 4)
            for act in DiscourseAct
        }

        # Calculate normalized context transitions P(Response | Context)
        transitions: Dict[str, Dict[str, float]] = {}
        for ctx_act in DiscourseAct:
            c_counts = transition_counts[ctx_act.value]
            total_c = sum(c_counts.values())
            transitions[ctx_act.value] = {
                resp_act.value: round(c_counts[resp_act.value] / total_c, 4) if total_c > 0 else 0.0
                for resp_act in DiscourseAct
            }

        # Top markers per act
        top_markers: Dict[str, List[Dict[str, Any]]] = {}
        for act in DiscourseAct:
            top_m = [
                {"marker": m, "count": cnt}
                for m, cnt in marker_counters[act.value].most_common(10)
            ]
            top_markers[act.value] = top_m

        return DiscourseProfile(
            total_turns=n_turns,
            act_distribution=act_dist,
            act_counts=dict(primary_counts),
            co_occurrence_matrix=co_occurrence,
            context_response_transitions=transitions,
            top_markers_by_act=top_markers,
            metadata={"name": name},
        )

    def analyze_by_style(
        self,
        pairs: Sequence[Dict[str, Any]],
        style_labels: Sequence[int],
        style_names: Optional[Dict[int, str]] = None,
    ) -> Dict[str, DiscourseProfile]:
        """
        Segments pairs by their communication archetype label and fits
        a DiscourseProfile for each archetype.
        """
        if len(pairs) != len(style_labels):
            raise ValueError("Pairs and style_labels must have the exact same length.")

        style_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for pair, label in zip(pairs, style_labels):
            style_groups[int(label)].append(pair)

        profiles: Dict[str, DiscourseProfile] = {}
        for label, group_pairs in sorted(style_groups.items()):
            name = (
                style_names.get(label, f"Style_{label}")
                if style_names
                else f"Style_{label}"
            )
            profiles[name] = self.fit(group_pairs, name=name)

        return profiles
