"""
Stage T037: Behavioral Alignment Evaluation.
Evaluates whether candidate responses align with the empirical behavioral dimensions
(hedging, directness, verbosity, disagreement, humor, confidence, formality)
established in the persona profile.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.nlp.behavior import BehavioralExtractor
from src.nlp.discourse import DiscourseAct, DiscourseAnalyzer

logger = logging.getLogger(__name__)

# Target profile default expectations derived from empirical global behavioral profile
DEFAULT_TARGET_BEHAVIOR = {
    "directness": 0.45,
    "verbosity": 0.31,
    "hedging": 0.03,
    "confidence": 0.48,
    "disagreement": 0.19,
    "humor": 0.22,
    "question": 0.28,
    "formality": 0.04,
    "code_switching": 0.48,
    "technical_depth": 0.38,
}

# Cues for excessive robotic formality or high hedging (anti-patterns)
HEDGING_CUES = re.compile(
    r"\b(i think that maybe|perhaps|i could be wrong|it seems to me|in my humble opinion|possibly|tentatively|i would guess)\b",
    re.IGNORECASE,
)
FORMALITY_CUES = re.compile(
    r"\b(dear|sincerely|regards|furthermore|moreover|consequently|nevertheless|pleasure|cordially|hereby|kindly|please be advised)\b",
    re.IGNORECASE,
)


@dataclass
class BehavioralMetricResult:
    """Quantitative behavioral alignment scores for a response."""

    candidate_dimensions: Dict[str, float]
    target_dimensions: Dict[str, float]
    mean_absolute_error: float       # [0.0, 1.0] lower is closer alignment
    cosine_alignment: float          # [0.0, 1.0] directional vector alignment
    hedging_penalty: float           # [0.0, 1.0] penalty for excessive hedging
    formality_penalty: float         # [0.0, 1.0] penalty for excessive formality
    composite_behavior_score: float  # [0.0, 1.0] overall behavioral alignment score

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["mean_absolute_error"] = round(self.mean_absolute_error, 4)
        res["cosine_alignment"] = round(self.cosine_alignment, 4)
        res["hedging_penalty"] = round(self.hedging_penalty, 4)
        res["formality_penalty"] = round(self.formality_penalty, 4)
        res["composite_behavior_score"] = round(self.composite_behavior_score, 4)
        res["candidate_dimensions"] = {k: round(v, 4) for k, v in self.candidate_dimensions.items()}
        res["target_dimensions"] = {k: round(v, 4) for k, v in self.target_dimensions.items()}
        return res


class BehavioralEvaluator:
    """
    Evaluates candidate responses against empirical behavioral trait benchmarks (T037).
    """

    def __init__(
        self,
        target_dimensions: Optional[Dict[str, float]] = None,
        behavior_extractor: Optional[BehavioralExtractor] = None,
    ):
        self.target_dimensions = target_dimensions or DEFAULT_TARGET_BEHAVIOR.copy()
        self.extractor = behavior_extractor or BehavioralExtractor()
        self.discourse_analyzer = self.extractor.discourse_analyzer

    def extract_turn_features(self, text: str) -> Dict[str, float]:
        """
        Extract behavioral feature values for a single response turn.
        """
        words = text.strip().split()
        n_words = len(words)
        if n_words == 0:
            return {k: 0.0 for k in self.target_dimensions}

        # 1. Directness: verbless fragments, brevity, absence of softeners
        is_fragment = 1.0 if n_words <= 3 else (0.5 if n_words <= 7 else 0.0)
        directness = is_fragment

        # 2. Verbosity: normalized word length
        verbosity = min(1.0, n_words / 30.0)

        # 3. Discourse analysis
        d_res = self.discourse_analyzer.analyze_turn(text)
        has_hedge = (DiscourseAct.HEDGE.value in d_res.act_tags) or bool(HEDGING_CUES.search(text))
        hedging = 1.0 if has_hedge else 0.0

        # 4. Disagreement style
        disagree = 1.0 if (DiscourseAct.DISAGREE.value in d_res.act_tags) else 0.0

        # 5. Humor / Banter
        humor = 1.0 if (DiscourseAct.HUMOR.value in d_res.act_tags) else 0.0

        # 6. Question
        question = 1.0 if ("?" in text or DiscourseAct.QUESTION.value in d_res.act_tags) else 0.0

        # 7. Confidence
        confidence = 1.0 if (d_res.primary_act in {DiscourseAct.INFORMATIVE.value, DiscourseAct.EXPLAIN.value, DiscourseAct.ADVICE.value} and not has_hedge) else 0.0

        # 8. Formality
        has_formal = bool(FORMALITY_CUES.search(text))
        formality = 1.0 if has_formal else 0.0

        # 9. Code-switching
        cs_res = self.extractor.cs_analyzer.analyze_turn(text)
        code_switching = 1.0 if cs_res.classification == "code_switched" else 0.0

        # 10. Technical depth
        re_tech = re.compile(r"\b(api|curl|endpoint|payload|json|server|backend|frontend|code|repo|branch|commit|bug|deploy|docker|python|wsl)\b", re.I)
        technical_depth = 1.0 if bool(re_tech.search(text)) else 0.0

        return {
            "directness": directness,
            "verbosity": verbosity,
            "hedging": hedging,
            "confidence": confidence,
            "disagreement": disagree,
            "humor": humor,
            "question": question,
            "formality": formality,
            "code_switching": code_switching,
            "technical_depth": technical_depth,
        }

    def evaluate_turn(
        self,
        candidate_text: str,
        target_reference_text: Optional[str] = None,
    ) -> BehavioralMetricResult:
        """
        Evaluate behavioral alignment of a single candidate response.
        """
        cand_dims = self.extract_turn_features(candidate_text)

        # Baseline targets from global profile
        target_dims = self.target_dimensions.copy()
        if target_reference_text:
            ref_dims = self.extract_turn_features(target_reference_text)
            # Blend 50% specific reference turn features with 50% global target traits
            target_dims = {k: 0.5 * target_dims.get(k, 0.0) + 0.5 * ref_dims.get(k, 0.0) for k in target_dims}

        keys = sorted(target_dims.keys())
        cand_vec = np.array([cand_dims.get(k, 0.0) for k in keys], dtype=np.float32)
        targ_vec = np.array([target_dims.get(k, 0.0) for k in keys], dtype=np.float32)

        # Mean Absolute Error
        mae = float(np.mean(np.abs(cand_vec - targ_vec)))

        # Cosine alignment
        norm_c = np.linalg.norm(cand_vec)
        norm_t = np.linalg.norm(targ_vec)
        if norm_c > 1e-6 and norm_t > 1e-6:
            cos_sim = float(np.dot(cand_vec, targ_vec) / (norm_c * norm_t))
        else:
            cos_sim = 1.0 - mae

        # Anti-pattern penalties: persona has near-zero hedging (0.03) and formality (0.04)
        # Any hedging or robotic formality in candidate incurs explicit penalty
        hedging_penalty = max(0.0, cand_dims.get("hedging", 0.0) - 0.05)
        formality_penalty = max(0.0, cand_dims.get("formality", 0.0) - 0.05)

        # Composite score
        base_score = max(0.0, 1.0 - mae)
        alignment_score = max(0.0, cos_sim)
        composite = 0.5 * base_score + 0.5 * alignment_score - (0.25 * hedging_penalty + 0.25 * formality_penalty)
        composite = min(1.0, max(0.0, composite))

        return BehavioralMetricResult(
            candidate_dimensions=cand_dims,
            target_dimensions=target_dims,
            mean_absolute_error=mae,
            cosine_alignment=cos_sim,
            hedging_penalty=hedging_penalty,
            formality_penalty=formality_penalty,
            composite_behavior_score=composite,
        )

    def evaluate_batch(
        self,
        candidates: Sequence[str],
        targets: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of candidate responses for behavioral alignment.
        """
        n = len(candidates)
        if n == 0:
            return {
                "count": 0,
                "mean_composite_score": 0.0,
                "mean_mae": 0.0,
                "mean_cosine_alignment": 0.0,
                "results": [],
            }

        results: List[BehavioralMetricResult] = []
        for i in range(n):
            cand = candidates[i]
            targ = targets[i] if targets and i < len(targets) else None
            res = self.evaluate_turn(cand, target_reference_text=targ)
            results.append(res)

        mean_composite = float(np.mean([r.composite_behavior_score for r in results]))
        mean_mae = float(np.mean([r.mean_absolute_error for r in results]))
        mean_cos = float(np.mean([r.cosine_alignment for r in results]))
        mean_hedge_pen = float(np.mean([r.hedging_penalty for r in results]))
        mean_formal_pen = float(np.mean([r.formality_penalty for r in results]))

        return {
            "count": n,
            "mean_composite_score": round(mean_composite, 4),
            "mean_mae": round(mean_mae, 4),
            "mean_cosine_alignment": round(mean_cos, 4),
            "mean_hedging_penalty": round(mean_hedge_pen, 4),
            "mean_formality_penalty": round(mean_formal_pen, 4),
            "results": [r.to_dict() for r in results],
        }
