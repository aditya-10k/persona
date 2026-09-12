"""
Evaluation Engine module (Stages T035-T040).
"""

from src.evaluation.behavior import (
    BehavioralEvaluator,
    BehavioralMetricResult,
)
from src.evaluation.engine import (
    CandidateEvaluationSummary,
    EvaluationEngine,
)
from src.evaluation.failure_analysis import (
    ERR_CASING_MISMATCH,
    ERR_DISAGREEMENT_FAILURE,
    ERR_EXCESSIVE_EMOJI,
    ERR_HALLUCINATED_PII,
    ERR_HIGH_HEDGING,
    ERR_ROBOTIC_AI,
    ERR_TOO_VERBOSE,
    ERR_TRAILING_PERIOD,
    ERR_WRONG_LANGUAGE_MIX,
    FailureAnalyzer,
    FailureReport,
    FailureTag,
)
from src.evaluation.judge import (
    DeterministicJudge,
    JudgeEvaluationResult,
    JudgeRubric,
)
from src.evaluation.metrics import (
    LinguisticEvaluator,
    LinguisticMetricResult,
)

__all__ = [
    "BehavioralEvaluator",
    "BehavioralMetricResult",
    "CandidateEvaluationSummary",
    "DeterministicJudge",
    "ERR_CASING_MISMATCH",
    "ERR_DISAGREEMENT_FAILURE",
    "ERR_EXCESSIVE_EMOJI",
    "ERR_HALLUCINATED_PII",
    "ERR_HIGH_HEDGING",
    "ERR_ROBOTIC_AI",
    "ERR_TOO_VERBOSE",
    "ERR_TRAILING_PERIOD",
    "ERR_WRONG_LANGUAGE_MIX",
    "EvaluationEngine",
    "FailureAnalyzer",
    "FailureReport",
    "FailureTag",
    "JudgeEvaluationResult",
    "JudgeRubric",
    "LinguisticEvaluator",
    "LinguisticMetricResult",
]
