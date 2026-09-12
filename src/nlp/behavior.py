"""
Stage T021: Behavioral Pattern Extraction.
Quantifies psychological and communicative behavioral dimensions (directness, verbosity,
hedging, disagreement style, humor, question initiative, confidence, expressiveness,
code-switching propensity, and technical depth) with supporting evidence exemplars.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.nlp.code_switching import CodeSwitchingAnalyzer
from src.nlp.discourse import DiscourseAct, DiscourseAnalyzer
from src.nlp.linguistics import LinguisticProfiler
from src.nlp.syntax import SyntacticProfiler

logger = logging.getLogger(__name__)


@dataclass
class BehaviorDimension:
    """A single quantified behavioral dimension with evidence score and exemplars."""

    name: str
    score: float             # Normalized [0.0, 1.0]
    confidence: float        # Statistical confidence [0.0, 1.0]
    method: str
    description: str
    supporting_exemplars: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "description": self.description,
            "supporting_exemplars": self.supporting_exemplars,
        }


@dataclass
class BehavioralProfile:
    """Comprehensive behavioral profile combining all 10 quantified dimensions."""

    total_turns: int
    dimensions: Dict[str, BehaviorDimension]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_turns": self.total_turns,
            "dimensions": {k: v.to_dict() for k, v in self.dimensions.items()},
            "metadata": self.metadata,
        }


class BehavioralExtractor:
    """
    Extracts 10 quantified behavioral dimensions from conversational turns
    by aggregating linguistic, syntactic, code-switching, and discourse metrics.
    """

    def __init__(self):
        self.ling_profiler = LinguisticProfiler()
        self.syntax_profiler = SyntacticProfiler()
        self.cs_analyzer = CodeSwitchingAnalyzer()
        self.discourse_analyzer = DiscourseAnalyzer()

    def extract_profile(
        self,
        pairs: Sequence[Dict[str, Any]],
        name: str = "global",
        max_exemplars: int = 5,
    ) -> BehavioralProfile:
        """
        Extracts the 10 behavioral dimensions across the provided turns.
        """
        n_turns = len(pairs)
        if n_turns == 0:
            raise ValueError("Pairs sequence cannot be empty for behavioral extraction.")

        # Aggregate turn-level feature scores
        directness_scores: List[Tuple[float, Dict[str, Any]]] = []
        verbosity_scores: List[Tuple[float, Dict[str, Any]]] = []
        hedging_scores: List[Tuple[float, Dict[str, Any]]] = []
        disagree_scores: List[Tuple[float, Dict[str, Any]]] = []
        humor_scores: List[Tuple[float, Dict[str, Any]]] = []
        question_scores: List[Tuple[float, Dict[str, Any]]] = []
        confidence_scores: List[Tuple[float, Dict[str, Any]]] = []
        expressiveness_scores: List[Tuple[float, Dict[str, Any]]] = []
        cs_scores: List[Tuple[float, Dict[str, Any]]] = []
        tech_scores: List[Tuple[float, Dict[str, Any]]] = []

        re_tech_words = re.compile(
            r"\b(api|curl|endpoint|payload|json|server|backend|frontend|code|repo|branch|commit|pr|merge|push|pull|bug|issue|deploy|docker|aws|sql|test|angular|react|node|python|java|ts|js|cdta|ach|air)\b",
            re.IGNORECASE,
        )

        for p in pairs:
            text = p.get("target_text", "").strip()
            ctx = p.get("context_text") or (p.get("context")[-1].get("text", "") if p.get("context") and isinstance(p.get("context")[-1], dict) else "")
            pid = p.get("pair_id", "")
            words = text.split()
            word_count = len(words)

            if word_count == 0:
                continue

            item_meta = {
                "pair_id": pid,
                "target_text": text,
                "context_text": ctx,
            }

            # 1. Directness: short punchy turns, verbless fragments, zero terminal punctuation, low hedging
            is_fragment = 1.0 if word_count <= 3 else (0.5 if word_count <= 7 else 0.0)
            d_score = is_fragment
            directness_scores.append((d_score, item_meta))

            # 2. Verbosity: sigmoid-scaled word count (normalized: 1-5 words ~ 0.2, 10 words ~ 0.5, 30+ words ~ 1.0)
            v_score = min(1.0, word_count / 30.0)
            verbosity_scores.append((v_score, item_meta))

            # 3. Hedging: presence of epistemic uncertainty
            d_res = self.discourse_analyzer.analyze_turn(text)
            h_score = 1.0 if DiscourseAct.HEDGE.value in d_res.act_tags else 0.0
            hedging_scores.append((h_score, item_meta))

            # 4. Disagreement style: blunt pushback
            dis_score = 1.0 if DiscourseAct.DISAGREE.value in d_res.act_tags else 0.0
            disagree_scores.append((dis_score, item_meta))

            # 5. Humor / Playfulness: laughter emojis and banter markers
            hum_score = 1.0 if DiscourseAct.HUMOR.value in d_res.act_tags else 0.0
            humor_scores.append((hum_score, item_meta))

            # 6. Question initiative: posing queries
            q_score = 1.0 if DiscourseAct.QUESTION.value in d_res.act_tags or "?" in text else 0.0
            question_scores.append((q_score, item_meta))

            # 7. Confidence & Assertion: declarative assertions without hesitation
            conf_score = 1.0 if (d_res.primary_act in {DiscourseAct.INFORMATIVE.value, DiscourseAct.EXPLAIN.value, DiscourseAct.ADVICE.value} and h_score == 0.0) else 0.0
            confidence_scores.append((conf_score, item_meta))

            # 8. Emotional expressiveness: emojis, exclamation, caps, intensifiers
            has_emojis = bool(re.search(r"[\U0001f000-\U0001f9ff]", text))
            has_emphasis = DiscourseAct.EMPHASIS.value in d_res.act_tags
            has_caps = text.isupper() and word_count > 1
            e_score = (1.0 if has_emojis else 0.0) * 0.5 + (1.0 if has_emphasis else 0.0) * 0.3 + (1.0 if has_caps else 0.0) * 0.2
            expressiveness_scores.append((e_score, item_meta))

            # 9. Code-switching propensity: Hinglish switching
            cs_res = self.cs_analyzer.analyze_turn(text)
            cs_score = min(1.0, cs_res.switch_points / 4.0) if cs_res.classification == "code_switched" else 0.0
            cs_scores.append((cs_score, item_meta))

            # 10. Technical depth: technical vocabulary concentration
            t_matches = re_tech_words.findall(text)
            tech_score = min(1.0, len(t_matches) / 3.0)
            tech_scores.append((tech_score, item_meta))

        def _make_dim(name: str, score_list: List[Tuple[float, Dict[str, Any]]], desc: str) -> BehaviorDimension:
            raw_scores = [s[0] for s in score_list]
            mean_score = float(np.mean(raw_scores)) if raw_scores else 0.0
            conf = min(1.0, len(raw_scores) / 500.0)  # High confidence with > 500 observations

            # Sort exemplars by highest score
            sorted_items = sorted(score_list, key=lambda x: x[0], reverse=True)
            top_exemplars = []
            seen_texts = set()
            for s, meta in sorted_items:
                if meta["target_text"] not in seen_texts and s > 0:
                    seen_texts.add(meta["target_text"])
                    top_exemplars.append({
                        "pair_id": meta["pair_id"],
                        "target_text": meta["target_text"],
                        "context_text": meta["context_text"],
                        "score": round(s, 2),
                    })
                    if len(top_exemplars) >= max_exemplars:
                        break

            return BehaviorDimension(
                name=name,
                score=round(mean_score, 4),
                confidence=round(conf, 4),
                method="statistical_distribution_extraction",
                description=desc,
                supporting_exemplars=top_exemplars,
            )

        dims = {
            "directness": _make_dim(
                "directness",
                directness_scores,
                "Tendency to respond with punchy, verbless fragments rather than circumlocutory prose.",
            ),
            "verbosity": _make_dim(
                "verbosity",
                verbosity_scores,
                "Relative message length and elaboration volume per conversational turn.",
            ),
            "hedging": _make_dim(
                "hedging",
                hedging_scores,
                "Frequency of epistemic uncertainty markers (shayad, lagra, maybe, i think).",
            ),
            "disagreement_style": _make_dim(
                "disagreement_style",
                disagree_scores,
                "Propensity for unvarnished, direct friction and denial (nahi re, chodd, galat).",
            ),
            "humor_playfulness": _make_dim(
                "humor_playfulness",
                humor_scores,
                "Density of laughter emojis, ironic sarcasm, and teasing slang.",
            ),
            "question_initiative": _make_dim(
                "question_initiative",
                question_scores,
                "Proactive conversational probing and follow-up inquiry frequency.",
            ),
            "confidence_assertion": _make_dim(
                "confidence_assertion",
                confidence_scores,
                "Declarative assertiveness without hedges or hesitation markers.",
            ),
            "emotional_expressiveness": _make_dim(
                "emotional_expressiveness",
                expressiveness_scores,
                "Dynamic emotional valence expressed via emojis, intensifiers, and casing.",
            ),
            "code_switching_propensity": _make_dim(
                "code_switching_propensity",
                cs_scores,
                "Frequency of intra-sentential language switches between English and Hindi.",
            ),
            "technical_depth": _make_dim(
                "technical_depth",
                tech_scores,
                "Density of programmatic vocabulary, API syntax, and engineering terminology.",
            ),
        }

        return BehavioralProfile(
            total_turns=n_turns,
            dimensions=dims,
            metadata={"name": name},
        )
