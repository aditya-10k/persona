"""
Unit tests for Evaluation Engine (Stages T035 - T040).
"""

import pytest
import numpy as np

from src.evaluation.behavior import BehavioralEvaluator, BehavioralMetricResult
from src.evaluation.engine import EvaluationEngine, CandidateEvaluationSummary
from src.evaluation.failure_analysis import (
    FailureAnalyzer,
    ERR_TRAILING_PERIOD,
    ERR_ROBOTIC_AI,
    ERR_TOO_VERBOSE,
    ERR_HIGH_HEDGING,
    ERR_HALLUCINATED_PII,
    ERR_EXCESSIVE_EMOJI,
)
from src.evaluation.judge import DeterministicJudge, JudgeRubric
from src.evaluation.metrics import LinguisticEvaluator, LinguisticMetricResult


class TestLinguisticEvaluator:
    @pytest.fixture
    def evaluator(self):
        # Initialized without heavy embedding engine for ultra-fast unit testing
        return LinguisticEvaluator(embedding_engine=None)

    def test_unpunctuated_rule_adherence(self, evaluator):
        # Clean unpunctuated candidate vs unpunctuated target
        res_clean = evaluator.evaluate_pair("ha sahi he bhai", "thik he karle")
        assert res_clean.punctuation_score == 1.0

        # Trailing period violation
        res_period = evaluator.evaluate_pair("ha sahi he bhai.", "thik he karle")
        assert res_period.punctuation_score == 0.0

    def test_casing_scoring(self, evaluator):
        # Both lowercase
        res_lower = evaluator.evaluate_pair("wsl hi use karra hu", "chodd na")
        assert res_lower.casing_score == 1.0

        # Shouting uppercase
        res_upper = evaluator.evaluate_pair("WSL HI USE KARRA HU ME BHAI", "chodd na")
        assert res_upper.casing_score < 0.5

    def test_length_divergence(self, evaluator):
        # Matching length
        res_match = evaluator.evaluate_pair("ha sahi he", "thik he chal")
        assert res_match.length_score >= 0.9

        # Excessive verbosity
        res_verbose = evaluator.evaluate_pair(
            "this is a very long candidate response with far too many unnecessary explanatory words that exceeds the brevity rule",
            "ok",
        )
        assert res_verbose.length_score < 0.5

    def test_batch_evaluation(self, evaluator):
        cands = ["ha sahi he", "nahi bhai galat bolra"]
        targs = ["haa sahi", "nahi aisa nahi he"]
        batch_res = evaluator.evaluate_batch(cands, targs)
        assert batch_res["count"] == 2
        assert 0.0 <= batch_res["mean_composite_score"] <= 1.0
        assert len(batch_res["results"]) == 2


class TestBehavioralEvaluator:
    @pytest.fixture
    def evaluator(self):
        return BehavioralEvaluator()

    def test_extract_turn_features(self, evaluator):
        # Direct, terse fragment
        feats_terse = evaluator.extract_turn_features("ha sahi he")
        assert feats_terse["directness"] >= 0.5
        assert feats_terse["hedging"] == 0.0
        assert feats_terse["formality"] == 0.0

        # Apologetic high hedging
        feats_hedge = evaluator.extract_turn_features("I think that maybe I could be wrong about this")
        assert feats_hedge["hedging"] == 1.0

    def test_hedging_and_formality_penalties(self, evaluator):
        res_authentic = evaluator.evaluate_turn("nahi bhai galat he")
        assert res_authentic.hedging_penalty == 0.0
        assert res_authentic.formality_penalty == 0.0

        res_robotic = evaluator.evaluate_turn("Dear friend, I think that maybe I could be mistaken.")
        assert res_robotic.hedging_penalty > 0.0
        assert res_robotic.formality_penalty > 0.0
        assert res_robotic.composite_behavior_score < res_authentic.composite_behavior_score

    def test_batch_behavioral_evaluation(self, evaluator):
        cands = ["ha bhai", "chodd wsl karta hu"]
        res = evaluator.evaluate_batch(cands)
        assert res["count"] == 2
        assert "mean_composite_score" in res


class TestJudgeAndRubric:
    def test_deterministic_judge_clean(self):
        res = DeterministicJudge.evaluate(
            real_response="chodd wsl hi karta hu",
            candidate_response="wsl use karle bhai",
            context="Friend: shrinking me issue aara",
        )
        assert res.overall_score >= 4.0
        assert res.stylistic_adherence == 5.0
        assert res.epistemic_grounding == 5.0

    def test_deterministic_judge_flags_violations(self):
        # Trailing period and robotic greeting
        res = DeterministicJudge.evaluate(
            real_response="ok",
            candidate_response="Certainly! I'd be happy to assist you with WSL.",
            context="Friend: shrinking issue",
        )
        assert res.stylistic_adherence < 4.0
        assert res.voice_authenticity == 1.0
        assert res.overall_score < 3.5

    def test_deterministic_judge_flags_pii_hallucination(self):
        res = DeterministicJudge.evaluate(
            real_response="pata nahi",
            candidate_response="mera password is secret123 aur phone is +919876543210",
            context="Friend: password kya he",
        )
        assert res.epistemic_grounding == 1.0

    def test_rubric_prompt_rendering_and_parsing(self):
        prompt = JudgeRubric.render_prompt(
            context="Friend: api run nahi ho rahi",
            real_response="curl chala ke dekh",
            candidate_response="curl check kar ek baar",
        )
        assert "voice_authenticity" in prompt
        assert "curl chala ke dekh" in prompt

        # Test parsing JSON judge response
        mock_json = '{"voice_authenticity": 4.5, "stylistic_adherence": 5.0, "behavioral_consistency": 4.0, "situational_appropriateness": 4.5, "epistemic_grounding": 5.0, "overall_score": 4.6, "reasoning": "Very natural Hinglish"}'
        parsed = JudgeRubric.parse_judge_response(mock_json)
        assert parsed.voice_authenticity == 4.5
        assert parsed.stylistic_adherence == 5.0
        assert parsed.overall_score == 4.6
        assert not parsed.is_offline_heuristic


class TestFailureAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return FailureAnalyzer()

    def test_detect_trailing_period(self, analyzer):
        tags = analyzer.analyze_turn("ha sahi he bhai.", target_text="ha sahi he")
        error_codes = [t.error_code for t in tags]
        assert ERR_TRAILING_PERIOD in error_codes

    def test_detect_robotic_assistant(self, analyzer):
        tags = analyzer.analyze_turn("Certainly! How can I assist you today?", target_text="ha")
        error_codes = [t.error_code for t in tags]
        assert ERR_ROBOTIC_AI in error_codes

    def test_detect_too_verbose(self, analyzer):
        verbose_text = " ".join(["word"] * 45)
        tags = analyzer.analyze_turn(verbose_text, target_text="ok")
        error_codes = [t.error_code for t in tags]
        assert ERR_TOO_VERBOSE in error_codes

    def test_detect_high_hedging(self, analyzer):
        tags = analyzer.analyze_turn("I might be wrong but maybe check it", target_text="check kar")
        error_codes = [t.error_code for t in tags]
        assert ERR_HIGH_HEDGING in error_codes

    def test_detect_pii_hallucination(self):
        analyzer = FailureAnalyzer()
        tags = analyzer.analyze_turn("mera phone number is +919876543210", target_text="idk")
        error_codes = [t.error_code for t in tags]
        assert ERR_HALLUCINATED_PII in error_codes

    def test_batch_failure_report(self, analyzer):
        cands = [
            "ha sahi he",                       # Clean
            "Certainly! I would be glad.",      # Robotic + Verbose
            "bhai check kar ek baar.",          # Trailing period
        ]
        targs = ["ha", "thik he", "check kar"]
        report = analyzer.analyze_batch(cands, targets=targs)
        assert report.total_turns == 3
        assert report.clean_turns == 1
        assert report.failed_turns == 2
        assert report.failure_rate > 0.5


class TestEvaluationEngine:
    def test_evaluate_strategy_and_markdown(self):
        engine = EvaluationEngine(
            linguistic_evaluator=LinguisticEvaluator(embedding_engine=None)
        )
        cands = ["ha sahi he", "wsl hi use karra hu", "nahi galat he"]
        targs = ["ha done", "wsl karta hu", "nahi aisa nahi"]
        contexts = ["api ready?", "shrinking issue", "branch merge hua?"]

        summary, details = engine.evaluate_strategy("test_strategy", cands, targs, contexts)
        assert summary.total_evaluated == 3
        assert 0.0 <= summary.linguistic_composite <= 1.0
        assert 1.0 <= summary.judge_overall <= 5.0

        md = engine.generate_markdown_report([summary])
        assert "# Persona Engine: Comprehensive Evaluation Report" in md
        assert "test_strategy" in md
