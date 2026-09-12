"""
Stage T040: Failure Analysis.
Systematically categorizes, counts, and extracts concrete failure modes
from candidate persona generation runs.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# Error Categories Taxonomy
ERR_TRAILING_PERIOD = "ERR_TRAILING_PERIOD"
ERR_ROBOTIC_AI = "ERR_ROBOTIC_AI"
ERR_TOO_VERBOSE = "ERR_TOO_VERBOSE"
ERR_HIGH_HEDGING = "ERR_HIGH_HEDGING"
ERR_EXCESSIVE_EMOJI = "ERR_EXCESSIVE_EMOJI"
ERR_WRONG_LANGUAGE_MIX = "ERR_WRONG_LANGUAGE_MIX"
ERR_HALLUCINATED_PII = "ERR_HALLUCINATED_PII"
ERR_CASING_MISMATCH = "ERR_CASING_MISMATCH"
ERR_DISAGREEMENT_FAILURE = "ERR_DISAGREEMENT_FAILURE"

ERROR_DESCRIPTIONS = {
    ERR_TRAILING_PERIOD: "Response ends with an unwanted full stop (.) violating the 92.5% unpunctuated rule.",
    ERR_ROBOTIC_AI: "Response contains canned corporate AI assistant phrases.",
    ERR_TOO_VERBOSE: "Response is excessively verbose compared to authentic terse messaging.",
    ERR_HIGH_HEDGING: "Response exhibits uncharacteristic apologetic or hedging qualifiers.",
    ERR_EXCESSIVE_EMOJI: "Response uses excessive emojis or non-characteristic decorative emojis.",
    ERR_WRONG_LANGUAGE_MIX: "Response fails to use natural Romanized Hinglish code-mixing.",
    ERR_HALLUCINATED_PII: "Response fabricates private real-world phone numbers, passwords, or credentials.",
    ERR_CASING_MISMATCH: "Response uses unwanted shouting uppercase or stiff formal title-case.",
    ERR_DISAGREEMENT_FAILURE: "Response fails to show expected direct skepticism/pushback in a conflict context.",
}

# Detection Regexes
RE_TRAILING_PERIOD = re.compile(r"\.\s*$")
RE_ROBOTIC_PHRASES = re.compile(
    r"\b(i'd be happy to|certainly|i hope this helps|feel free to|let me know if you need|as an artificial intelligence|as an ai|how can i assist|is there anything else)\b",
    re.I,
)
RE_HEDGING_PHRASES = re.compile(
    r"\b(i might be wrong|in my humble opinion|it seems to me|perhaps|tentatively|i could be mistaken)\b",
    re.I,
)
RE_PII = re.compile(
    r"((?:\+91[\s-]?)?[6-9]\d{9}|(?:\+91\d{10})|password\s*(?:is|=|:)|phone\s*(?:number\s*)?(?:is|=|:)|[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",
    re.I,
)
RE_EMOJI = re.compile(r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf]")
RE_FORBIDDEN_EMOJI = re.compile(r"[🎉✨🤖😊👍👋]")


@dataclass
class FailureTag:
    """Individual failure detected in a candidate response."""

    error_code: str
    description: str
    snippet: str
    severity: str  # "high", "medium", "low"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FailureReport:
    """Aggregated failure report across an evaluation batch."""

    total_turns: int
    failed_turns: int
    clean_turns: int
    failure_rate: float
    error_counts: Dict[str, int]
    error_rates: Dict[str, float]
    examples_by_error: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["failure_rate"] = round(self.failure_rate, 4)
        res["error_rates"] = {k: round(v, 4) for k, v in self.error_rates.items()}
        return res


class FailureAnalyzer:
    """
    Analyzes candidate responses for concrete persona violations and produces
    structured diagnostic reports (T040).
    """

    def analyze_turn(
        self,
        candidate_text: str,
        target_text: str = "",
        context_text: str = "",
    ) -> List[FailureTag]:
        """
        Inspect a single candidate response and return all detected failures.
        """
        cand = candidate_text.strip()
        words = cand.split()
        n_words = len(words)
        failures: List[FailureTag] = []

        # 1. Trailing Period
        if RE_TRAILING_PERIOD.search(cand):
            # Only flag if target didn't also end with period
            if not RE_TRAILING_PERIOD.search(target_text.strip()):
                failures.append(
                    FailureTag(
                        error_code=ERR_TRAILING_PERIOD,
                        description=ERROR_DESCRIPTIONS[ERR_TRAILING_PERIOD],
                        snippet=cand[-5:],
                        severity="medium",
                    )
                )

        # 2. Robotic AI Assistant Marker
        match_ai = RE_ROBOTIC_PHRASES.search(cand)
        if match_ai:
            failures.append(
                FailureTag(
                    error_code=ERR_ROBOTIC_AI,
                    description=ERROR_DESCRIPTIONS[ERR_ROBOTIC_AI],
                    snippet=match_ai.group(0),
                    severity="high",
                )
            )

        # 3. Excessive Verbosity
        target_words = len(target_text.strip().split())
        if n_words > 35 or (target_words > 0 and n_words > max(15, int(target_words * 2.5))):
            failures.append(
                FailureTag(
                    error_code=ERR_TOO_VERBOSE,
                    description=f"{ERROR_DESCRIPTIONS[ERR_TOO_VERBOSE]} ({n_words} words vs target {target_words} words)",
                    snippet=cand[:50] + "...",
                    severity="high" if n_words > 40 else "medium",
                )
            )

        # 4. High Hedging
        match_hedge = RE_HEDGING_PHRASES.search(cand)
        if match_hedge:
            failures.append(
                FailureTag(
                    error_code=ERR_HIGH_HEDGING,
                    description=ERROR_DESCRIPTIONS[ERR_HIGH_HEDGING],
                    snippet=match_hedge.group(0),
                    severity="medium",
                )
            )

        # 5. Excessive / Uncharacteristic Emojis
        emojis_found = RE_EMOJI.findall(cand)
        has_forbidden_emoji = bool(RE_FORBIDDEN_EMOJI.search(cand))
        if len(emojis_found) > 3 or has_forbidden_emoji:
            failures.append(
                FailureTag(
                    error_code=ERR_EXCESSIVE_EMOJI,
                    description=ERROR_DESCRIPTIONS[ERR_EXCESSIVE_EMOJI],
                    snippet="".join(emojis_found),
                    severity="low",
                )
            )

        # 6. Hallucinated PII / Credentials
        match_pii = RE_PII.search(cand)
        if match_pii:
            failures.append(
                FailureTag(
                    error_code=ERR_HALLUCINATED_PII,
                    description=ERROR_DESCRIPTIONS[ERR_HALLUCINATED_PII],
                    snippet=match_pii.group(0),
                    severity="high",
                )
            )

        # 7. Casing Mismatch (shouting caps)
        if cand.isupper() and n_words > 2:
            failures.append(
                FailureTag(
                    error_code=ERR_CASING_MISMATCH,
                    description=ERROR_DESCRIPTIONS[ERR_CASING_MISMATCH],
                    snippet=cand[:30],
                    severity="medium",
                )
            )

        # 8. Wrong Language Mix (Pure formal English in casual situation with no tech context)
        if n_words > 4 and not any(c in cand.lower() for c in ["bhai", "nahi", "ha", "toh", "ka", "ki", "ke", "me", "he", "ho", "kya", "kar"]):
            if not any(tech in cand.lower() or tech in context_text.lower() for tech in ["api", "curl", "json", "endpoint", "wsl", "docker", "server"]):
                failures.append(
                    FailureTag(
                        error_code=ERR_WRONG_LANGUAGE_MIX,
                        description=ERROR_DESCRIPTIONS[ERR_WRONG_LANGUAGE_MIX],
                        snippet=cand[:40],
                        severity="low",
                    )
                )

        return failures

    def analyze_batch(
        self,
        candidates: Sequence[str],
        targets: Optional[Sequence[str]] = None,
        contexts: Optional[Sequence[str]] = None,
        max_examples_per_error: int = 3,
    ) -> FailureReport:
        """
        Analyze an entire evaluation run and generate an aggregated failure report.
        """
        n = len(candidates)
        if n == 0:
            return FailureReport(
                total_turns=0,
                failed_turns=0,
                clean_turns=0,
                failure_rate=0.0,
                error_counts={},
                error_rates={},
                examples_by_error={},
            )

        error_counts: Dict[str, int] = defaultdict(int)
        examples_by_error: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        failed_turns_set = set()

        for i in range(n):
            cand = candidates[i]
            targ = targets[i] if targets and i < len(targets) else ""
            ctx = contexts[i] if contexts and i < len(contexts) else ""

            tags = self.analyze_turn(cand, target_text=targ, context_text=ctx)
            if tags:
                failed_turns_set.add(i)
                for tag in tags:
                    error_counts[tag.error_code] += 1
                    if len(examples_by_error[tag.error_code]) < max_examples_per_error:
                        examples_by_error[tag.error_code].append({
                            "candidate": cand,
                            "target": targ,
                            "snippet": tag.snippet,
                            "severity": tag.severity,
                        })

        failed_count = len(failed_turns_set)
        clean_count = n - failed_count
        fail_rate = failed_count / n if n > 0 else 0.0
        error_rates = {k: v / n for k, v in error_counts.items()}

        return FailureReport(
            total_turns=n,
            failed_turns=failed_count,
            clean_turns=clean_count,
            failure_rate=fail_rate,
            error_counts=dict(error_counts),
            error_rates=error_rates,
            examples_by_error=dict(examples_by_error),
        )
