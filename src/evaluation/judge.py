"""
Stage T038 & T039: LLM-as-Judge and Epistemic Boundary Evaluation.
Provides structured multi-axis evaluation prompts, parsing logic, and
a deterministic offline heuristic judge for regression testing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEMPLATE = """You are an expert conversational forensic evaluator.
Your task is to evaluate an AI candidate response intended to clone a specific human communication persona.

### CONTEXT & REFERENCE:
Conversation History:
{context}

Authentic Real Human Response (Ground Truth):
{real_response}

Generated AI Candidate Response:
{candidate_response}

---

### EVALUATION RUBRIC (Score each axis from 1 to 5):

1. **voice_authenticity** (1=completely fake/robotic, 5=indistinguishable from authentic Hinglish speaker):
   - Does it sound like an authentic Romanized Hinglish speaker?
   - Is the phrasing natural or stiffly translated?

2. **stylistic_adherence** (1=violates all rules, 5=flawless stylistic fidelity):
   - NO trailing period at the end of the message.
   - Pacing: Brief, punchy, conversational (no multi-paragraph essays).
   - Casing: Defaults to lowercase in casual chats, mixed case in technical context.

3. **behavioral_consistency** (1=wrong persona, 5=perfect persona match):
   - Does it reflect low hedging (no timid apologies, no "I might be mistaken")?
   - Is it direct and unpretentious?

4. **situational_appropriateness** (1=completely inappropriate register, 5=exact match):
   - Does it match the technical, casual, or conflict register required by the context?

5. **epistemic_grounding** (1=hallucinates private facts/passwords/names, 5=strictly grounded or defers naturally):
   - Does it avoid making up unknown private real-world facts?
   - If asked about secrets or unknown facts, does it defer naturally (e.g. "pata nahi bhai")?

---

### OUTPUT FORMAT:
Output ONLY valid JSON with no preamble or code fences:
{{
  "voice_authenticity": <1-5>,
  "stylistic_adherence": <1-5>,
  "behavioral_consistency": <1-5>,
  "situational_appropriateness": <1-5>,
  "epistemic_grounding": <1-5>,
  "overall_score": <1.0-5.0>,
  "reasoning": "<brief 1-2 sentence justification>"
}}
"""


@dataclass
class JudgeEvaluationResult:
    """Structured result of an LLM-as-judge evaluation."""

    voice_authenticity: float          # 1.0 - 5.0
    stylistic_adherence: float         # 1.0 - 5.0
    behavioral_consistency: float      # 1.0 - 5.0
    situational_appropriateness: float # 1.0 - 5.0
    epistemic_grounding: float         # 1.0 - 5.0
    overall_score: float               # 1.0 - 5.0
    reasoning: str = ""
    is_offline_heuristic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voice_authenticity": round(self.voice_authenticity, 2),
            "stylistic_adherence": round(self.stylistic_adherence, 2),
            "behavioral_consistency": round(self.behavioral_consistency, 2),
            "situational_appropriateness": round(self.situational_appropriateness, 2),
            "epistemic_grounding": round(self.epistemic_grounding, 2),
            "overall_score": round(self.overall_score, 2),
            "reasoning": self.reasoning,
            "is_offline_heuristic": self.is_offline_heuristic,
        }


class JudgeRubric:
    """
    Manages evaluation prompt rendering, response parsing, and fallback evaluation.
    """

    @staticmethod
    def render_prompt(
        context: str,
        real_response: str,
        candidate_response: str,
    ) -> str:
        """Render the structured evaluation prompt."""
        return JUDGE_PROMPT_TEMPLATE.format(
            context=context.strip() or "(No prior context)",
            real_response=real_response.strip(),
            candidate_response=candidate_response.strip(),
        )

    @staticmethod
    def parse_judge_response(raw_text: str) -> JudgeEvaluationResult:
        """Parse structured JSON from an LLM judge output."""
        cleaned = raw_text.strip()
        # Remove code blocks if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
            v = float(data.get("voice_authenticity", 3.0))
            s = float(data.get("stylistic_adherence", 3.0))
            b = float(data.get("behavioral_consistency", 3.0))
            sit = float(data.get("situational_appropriateness", 3.0))
            e = float(data.get("epistemic_grounding", 5.0))
            overall = float(data.get("overall_score", (v + s + b + sit + e) / 5.0))
            reasoning = str(data.get("reasoning", ""))
            return JudgeEvaluationResult(
                voice_authenticity=min(5.0, max(1.0, v)),
                stylistic_adherence=min(5.0, max(1.0, s)),
                behavioral_consistency=min(5.0, max(1.0, b)),
                situational_appropriateness=min(5.0, max(1.0, sit)),
                epistemic_grounding=min(5.0, max(1.0, e)),
                overall_score=min(5.0, max(1.0, overall)),
                reasoning=reasoning,
                is_offline_heuristic=False,
            )
        except Exception as err:
            logger.warning("Failed to parse LLM judge response JSON: %s. Falling back to heuristic.", err)
            return DeterministicJudge.evaluate(
                real_response="",
                candidate_response=cleaned,
                context="",
            )


class DeterministicJudge:
    """
    Offline deterministic heuristic judge that scores candidates across the 5 rubric axes
    using rule-based checks, stylistic constraints, and epistemic boundary inspection.
    """

    RE_ROBOTIC_AI = re.compile(
        r"\b(i'd be happy to|certainly|how can i assist|as an ai|i hope this helps|feel free to|let me know if you need)\b",
        re.I,
    )
    RE_TRAILING_PERIOD = re.compile(r"\.\s*$")
    RE_HEDGING = re.compile(r"\b(i might be wrong|perhaps|i think maybe|in my humble opinion)\b", re.I)
    RE_PII_HALLUCINATION = re.compile(r"\b(\+91\d{10}|\d{10}|password\s*is|my\s*phone\s*is)\b", re.I)

    @classmethod
    def evaluate(
        cls,
        real_response: str,
        candidate_response: str,
        context: str = "",
    ) -> JudgeEvaluationResult:
        cand = candidate_response.strip()
        reasons = []

        # 1. Stylistic Adherence (1 to 5)
        style_score = 5.0
        if cls.RE_TRAILING_PERIOD.search(cand) and not cls.RE_TRAILING_PERIOD.search(real_response):
            style_score -= 1.5
            reasons.append("unwanted trailing period")
        if len(cand.split()) > 35:
            style_score -= 1.5
            reasons.append("too verbose")
        if cand.isupper() and len(cand.split()) > 2:
            style_score -= 1.0
            reasons.append("excessive uppercase")
        style_score = max(1.0, style_score)

        # 2. Voice Authenticity (1 to 5)
        voice_score = 5.0
        behavior_score = 5.0
        if cls.RE_ROBOTIC_AI.search(cand):
            voice_score = 1.0
            behavior_score = 1.5
            reasons.append("canned AI assistant pleasantries")
        else:
            # Check Hinglish naturalness
            words = set(cand.lower().split())
            hinglish_markers = {"bhai", "ha", "nahi", "kar", "he", "ho", "tu", "me", "bc", "sahi", "kya", "toh", "ka", "ki", "ko"}
            has_hinglish = bool(words.intersection(hinglish_markers))
            if not has_hinglish and len(words) > 3:
                voice_score -= 1.5
                reasons.append("pure English without natural code-mixing")
        voice_score = max(1.0, voice_score)

        # 3. Behavioral Consistency (1 to 5)
        if cls.RE_HEDGING.search(cand):
            behavior_score = min(behavior_score, 3.0)
            behavior_score -= 1.5
            reasons.append("excessive uncharacteristic hedging")
        behavior_score = max(1.0, behavior_score)

        # 4. Situational Appropriateness (1 to 5)
        sit_score = 4.5
        # If context had technical cues, candidate should have technical cues or brevity
        if any(tech in context.lower() for tech in ["api", "curl", "endpoint", "bug", "code"]):
            if any(tech in cand.lower() for tech in ["api", "curl", "endpoint", "wsl", "code", "run", "check", "host"]):
                sit_score = 5.0
            elif len(cand.split()) <= 4:
                sit_score = 4.5
            else:
                sit_score = 3.5

        # 5. Epistemic Grounding (1 to 5)
        epistemic_score = 5.0
        if cls.RE_PII_HALLUCINATION.search(cand):
            epistemic_score = 1.0
            reasons.append("hallucinated private PII")
        elif "password" in context.lower() or "phone" in context.lower():
            if any(w in cand.lower() for w in ["pata nahi", "idk", "nahi", "kya"]):
                epistemic_score = 5.0
            else:
                epistemic_score = 3.0

        overall = (voice_score + style_score + behavior_score + sit_score + epistemic_score) / 5.0
        reasoning_str = "; ".join(reasons) if reasons else "Adheres cleanly to style, voice, and behavioral constraints"

        return JudgeEvaluationResult(
            voice_authenticity=round(voice_score, 2),
            stylistic_adherence=round(style_score, 2),
            behavioral_consistency=round(behavior_score, 2),
            situational_appropriateness=round(sit_score, 2),
            epistemic_grounding=round(epistemic_score, 2),
            overall_score=round(overall, 2),
            reasoning=reasoning_str,
            is_offline_heuristic=True,
        )
