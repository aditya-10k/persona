"""
Stage T009: Quality Filtering

Filters out non-linguistic noise (media-only placeholders, pure URL drops, empty responses,
punctuation pings, system notices) while strictly preserving:
- Short stylistic reactions ("ha", "hmm", "nah", "yes", "ok", "scam", "bro")
- Internet slang ("lol", "lmao", "rofl", "ngl", "tbh")
- Solo emojis ("💀", "😭", "🔥")
- Multi-modal commentary (text sent alongside media or URLs)
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import emoji

logger = logging.getLogger(__name__)


class QualityFilter:
    """
    Applies deterministic quality rules to context-response pairs.
    """

    # Pure media placeholders (standalone or multiple omitted media tags)
    MEDIA_ONLY_PATTERN = re.compile(
        r'(?i)^(?:\[Forwarded\]\s*)?(?:<[^>]+omitted>\s*)+$'
    )

    # Pure URL drops with no text commentary
    PURE_URL_PATTERN = re.compile(
        r'^(?:\[Forwarded\]\s*)?\[URL\]$'
    )

    # Punctuation-only pings (e.g. "..", "...", ".", "--") with no alphanumeric characters or emojis
    PUNCT_PING_PATTERN = re.compile(
        r'^[.\s,;:\-_*~`\'"^/\\|]+$'
    )

    # WhatsApp system notification patterns
    SYSTEM_EVENT_PATTERN = re.compile(
        r'(?i)^-\s*(?:Messages and calls are end-to-end encrypted|Missed voice call|Missed video call|You added|You removed|You created|You changed)'
    )

    def evaluate_pair(self, pair: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Evaluates a context-response pair.
        Returns:
            (is_valid: bool, drop_reason: Optional[str])
        """
        target_text = pair.get("target_text", "").strip()
        target_analysis = pair.get("target_analysis_text", "").strip()

        # 1. Empty or whitespace-only response
        if not target_text:
            return False, "empty_response"

        # 2. Pure media placeholder (e.g. <image omitted>, <GIF omitted>)
        if self.MEDIA_ONLY_PATTERN.match(target_text) or (
            target_text.startswith("<") and target_text.endswith("omitted>")
        ):
            return False, "pure_media_response"

        # 3. Pure URL drop without commentary
        if self.PURE_URL_PATTERN.match(target_text):
            return False, "pure_url_response"

        # 4. WhatsApp system notifications or call notices
        if self.SYSTEM_EVENT_PATTERN.match(target_text):
            return False, "system_event_response"

        # 5. Punctuation pings (dots, dashes with zero words or emojis)
        has_emoji = emoji.emoji_count(target_text) > 0
        if not has_emoji and self.PUNCT_PING_PATTERN.match(target_text):
            return False, "punctuation_ping"

        # 6. Analysis text is empty (e.g. boilerplate stripped everything away)
        if not target_analysis:
            return False, "empty_analysis_text"

        # All other responses (including short words "ha", "hmm", "💀") are VALID
        return True, None


def filter_context_pairs_dataset(
    input_pairs_path: str,
    output_pairs_path: str,
    report_output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Streams context_pairs.jsonl, filters uninformative noise,
    and writes filtered_pairs.jsonl and quality_filtering_report.json.
    """
    in_path = Path(input_pairs_path)
    out_path = Path(output_pairs_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    filter_engine = QualityFilter()

    total_pairs = 0
    passed_pairs = 0
    filtered_pairs = 0

    drop_reasons = Counter()
    passed_by_depth = Counter()
    passed_by_file = Counter()
    passed_initiations = 0
    passed_replies = 0

    with open(in_path, "r", encoding="utf-8") as in_f, open(
        out_path, "w", encoding="utf-8"
    ) as out_f:
        for line in in_f:
            line_str = line.strip()
            if not line_str:
                continue

            pair = json.loads(line_str)
            total_pairs += 1

            is_valid, drop_reason = filter_engine.evaluate_pair(pair)

            if is_valid:
                passed_pairs += 1
                pair_record = dict(pair)
                pair_record["quality_status"] = "PASSED"
                out_f.write(json.dumps(pair_record, ensure_ascii=False) + "\n")

                if pair.get("is_initiation"):
                    passed_initiations += 1
                else:
                    passed_replies += 1

                passed_by_depth[pair.get("context_depth", 0)] += 1
                passed_by_file[pair.get("source_file", "unknown")] += 1
            else:
                filtered_pairs += 1
                drop_reasons[drop_reason or "unknown"] += 1

    report = {
        "schema_version": "1.0",
        "task_id": "T009",
        "status": "COMPLETE",
        "total_pairs_processed": total_pairs,
        "total_pairs_passed": passed_pairs,
        "total_pairs_filtered": filtered_pairs,
        "pass_rate_percentage": round(passed_pairs / total_pairs * 100, 2) if total_pairs else 0,
        "filter_reasons_breakdown": dict(drop_reasons),
        "passed_pairs_metrics": {
            "initiation_pairs": passed_initiations,
            "reply_pairs": passed_replies,
            "by_context_depth": {
                f"{k}_turns": passed_by_depth[k] for k in sorted(passed_by_depth.keys())
            },
            "by_source_file": dict(passed_by_file),
        },
    }

    if report_output_path:
        rep_path = Path(report_output_path)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rep_path, "w", encoding="utf-8") as rep_f:
            json.dump(report, rep_f, indent=2, ensure_ascii=False)

    return report
