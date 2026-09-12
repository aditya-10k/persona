"""
Unit and integration tests for T009 Quality Filtering.
"""

import json
from pathlib import Path
import pytest

from src.dataset.quality_filter import QualityFilter, filter_context_pairs_dataset


@pytest.fixture
def quality_filter():
    return QualityFilter()


def test_filters_pure_media_placeholders(quality_filter):
    bad_pairs = [
        {"target_text": "<image omitted>", "target_analysis_text": ""},
        {"target_text": "<GIF omitted>", "target_analysis_text": ""},
        {"target_text": "<sticker omitted>", "target_analysis_text": ""},
        {"target_text": "[Forwarded] <image omitted>", "target_analysis_text": ""},
        {"target_text": "<image omitted>\n<image omitted>", "target_analysis_text": ""},
    ]
    for p in bad_pairs:
        is_valid, reason = quality_filter.evaluate_pair(p)
        assert is_valid is False
        assert reason in ("pure_media_response", "empty_analysis_text")


def test_filters_pure_url_drops(quality_filter):
    p = {"target_text": "[URL]", "target_analysis_text": "[URL]"}
    is_valid, reason = quality_filter.evaluate_pair(p)
    assert is_valid is False
    assert reason == "pure_url_response"

    p2 = {"target_text": "[Forwarded] [URL]", "target_analysis_text": "[URL]"}
    is_valid2, reason2 = quality_filter.evaluate_pair(p2)
    assert is_valid2 is False
    assert reason2 == "pure_url_response"


def test_preserves_text_accompanying_media_and_urls(quality_filter):
    p1 = {
        "target_text": "look at this photo\n<image omitted>",
        "target_analysis_text": "look at this photo",
    }
    is_valid1, reason1 = quality_filter.evaluate_pair(p1)
    assert is_valid1 is True
    assert reason1 is None

    p2 = {
        "target_text": "bro see this reel [URL] 💀",
        "target_analysis_text": "bro see this reel [URL] 💀",
    }
    is_valid2, reason2 = quality_filter.evaluate_pair(p2)
    assert is_valid2 is True
    assert reason2 is None


def test_critical_preservation_of_short_responses(quality_filter):
    """
    CRITICAL: Verify that short human stylistic answers are NOT filtered out!
    """
    short_answers = [
        "ha",
        "hmm",
        "yes",
        "nah",
        "bro",
        "ok",
        "lol",
        "lmao",
        "scam",
        "true",
        "💀",
        "😭",
        "🙏",
        "WHAT?!",
    ]
    for ans in short_answers:
        pair = {"target_text": ans, "target_analysis_text": ans}
        is_valid, reason = quality_filter.evaluate_pair(pair)
        assert is_valid is True, f"Falsely filtered short answer: {ans} (reason: {reason})"
        assert reason is None


def test_filters_punctuation_pings(quality_filter):
    p1 = {"target_text": "..", "target_analysis_text": ".."}
    is_valid1, reason1 = quality_filter.evaluate_pair(p1)
    assert is_valid1 is False
    assert reason1 == "punctuation_ping"

    p2 = {"target_text": "...", "target_analysis_text": "..."}
    is_valid2, reason2 = quality_filter.evaluate_pair(p2)
    assert is_valid2 is False
    assert reason2 == "punctuation_ping"


def test_filter_dataset_streaming(tmp_path):
    input_file = tmp_path / "context_pairs.jsonl"
    output_file = tmp_path / "filtered_pairs.jsonl"
    report_file = tmp_path / "quality_report.json"

    pairs = [
        # Valid reply
        {
            "pair_id": "p1",
            "source_file": "chat.txt",
            "is_initiation": False,
            "context_depth": 1,
            "target_text": "ha mai karta hu",
            "target_analysis_text": "ha mai karta hu",
        },
        # Valid short emoji reply
        {
            "pair_id": "p2",
            "source_file": "chat.txt",
            "is_initiation": False,
            "context_depth": 1,
            "target_text": "💀",
            "target_analysis_text": "💀",
        },
        # Pure media drop
        {
            "pair_id": "p3",
            "source_file": "chat.txt",
            "is_initiation": False,
            "context_depth": 1,
            "target_text": "<image omitted>",
            "target_analysis_text": "",
        },
        # Pure URL drop
        {
            "pair_id": "p4",
            "source_file": "chat.txt",
            "is_initiation": False,
            "context_depth": 1,
            "target_text": "[URL]",
            "target_analysis_text": "[URL]",
        },
    ]

    with open(input_file, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    report = filter_context_pairs_dataset(str(input_file), str(output_file), str(report_file))

    assert report["total_pairs_processed"] == 4
    assert report["total_pairs_passed"] == 2
    assert report["total_pairs_filtered"] == 2
    assert report["filter_reasons_breakdown"]["pure_media_response"] == 1
    assert report["filter_reasons_breakdown"]["pure_url_response"] == 1

    with open(output_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]

    assert len(lines) == 2
    assert lines[0]["pair_id"] == "p1"
    assert lines[0]["quality_status"] == "PASSED"
    assert lines[1]["pair_id"] == "p2"
    assert lines[1]["target_text"] == "💀"
