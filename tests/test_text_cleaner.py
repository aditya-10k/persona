"""
Unit and integration tests for T006 Text Cleaning & Multi-View Normalization.
"""

import json
from pathlib import Path
import pytest

from src.preprocessing.text_cleaner import TextCleaner, clean_dataset


@pytest.fixture
def cleaner():
    return TextCleaner()


def test_unicode_and_whitespace_normalization(cleaner):
    # Text with zero-width space, non-breaking space, windows newline, excessive horizontal space
    raw = "Hello\u200b   world\xa0there!\r\nHow   are\t\tyou?"
    normalized = cleaner.normalize_text(raw)
    assert "\u200b" not in normalized
    assert "\xa0" not in normalized
    assert "\r" not in normalized
    assert "Hello world there!\nHow are you?" == normalized


def test_preserves_casing(cleaner):
    raw = "broooo this is FUCKED 😭😭"
    views = cleaner.create_views(raw)

    # Casing must NOT be lowercased in normalized_text or analysis_text
    assert "FUCKED" in views["normalized_text"]
    assert "FUCKED" in views["analysis_text"]
    assert "fucked" not in views["normalized_text"]
    assert "fucked" not in views["analysis_text"]


def test_controlled_elongation_reduction(cleaner):
    raw = "brooooo that is sooooo coollll"
    views = cleaner.create_views(raw)

    # raw and normalized retain original elongations
    assert views["raw_text"] == raw
    assert views["normalized_text"] == raw

    # analysis_text reduces 3+ identical characters down to 2
    assert views["analysis_text"] == "broo that is soo cooll"


def test_emoji_and_punctuation_spacing_in_analysis(cleaner):
    raw = "done😭really???cool"
    views = cleaner.create_views(raw)

    # Emoji and punct clusters detached with spaces for subword tokenizers
    assert views["analysis_text"] == "done 😭 really ??? cool"


def test_whatsapp_boilerplate_stripping_in_analysis(cleaner):
    raw = "[Forwarded] Check this out: <image omitted>"
    views = cleaner.create_views(raw)

    assert "[Forwarded]" in views["raw_text"]
    assert "<image omitted>" in views["raw_text"]

    # Stripped from analysis_text
    assert "[Forwarded]" not in views["analysis_text"]
    assert "<image omitted>" not in views["analysis_text"]
    assert views["analysis_text"] == "Check this out:"


def test_style_marker_extraction(cleaner):
    raw = "broooo this is FUCKED 😭😭??? Are you serious?!"
    views = cleaner.create_views(raw)
    style = views["tokenized_text"]["style_markers"]

    # Emojis
    assert style["has_emoji"] is True
    assert style["emojis"] == ["😭", "😭"]
    assert style["emoji_count"] == 2

    # Elongation
    assert style["has_elongation"] is True
    assert len(style["elongated_words"]) == 1
    assert style["elongated_words"][0]["word"] == "broooo"
    assert style["elongated_words"][0]["base"] == "broo"
    assert style["elongated_words"][0]["char"] == "o"
    assert style["elongated_words"][0]["count"] == 4

    # All caps
    assert style["has_all_caps"] is True
    assert "FUCKED" in style["all_caps_words"]
    # Single letter or masked tokens shouldn't be counted
    assert "I" not in style["all_caps_words"]

    # Punctuation clusters
    assert style["has_punct_cluster"] is True
    assert "???" in style["punctuation_clusters"]
    assert "?!" in style["punctuation_clusters"]


def test_clean_record_preserves_sender_and_provenance(cleaner):
    record = {
        "message_id": "test:001",
        "timestamp": "2026-05-16T18:38:11",
        "sender": "+91 91234 56789",
        "text": "broooo [URL] 💀",
        "message_type": "text",
        "is_system": False,
        "is_media": False,
        "is_forwarded": False,
        "source_file": "test.txt",
        "raw_line_start": 4,
        "raw_line_end": 4,
        "parse_warnings": [],
    }

    clean = cleaner.clean_record(record)

    # Sender strictly unchanged
    assert clean["sender"] == "+91 91234 56789"
    assert clean["message_id"] == "test:001"
    assert clean["source_file"] == "test.txt"

    # Views present
    assert "views" in clean
    assert clean["views"]["raw_text"] == "broooo [URL] 💀"
    assert clean["views"]["analysis_text"] == "broo [URL] 💀"
    assert clean["views"]["tokenized_text"]["style_markers"]["has_emoji"] is True


def test_clean_dataset_streaming(tmp_path):
    input_file = tmp_path / "sanitized_messages.jsonl"
    output_file = tmp_path / "cleaned_messages.jsonl"
    report_file = tmp_path / "cleaning_report.json"

    records = [
        {
            "message_id": "t:1",
            "timestamp": "2026-05-16T18:38:11",
            "sender": "You",
            "text": "broooo this is FUCKED 😭😭",
            "message_type": "text",
            "is_system": False,
            "is_media": False,
            "is_forwarded": False,
            "source_file": "test.txt",
            "raw_line_start": 1,
            "raw_line_end": 1,
            "parse_warnings": [],
        },
        {
            "message_id": "t:2",
            "timestamp": "2026-05-16T18:38:17",
            "sender": "Friend_01",
            "text": "Normal text here",
            "message_type": "text",
            "is_system": False,
            "is_media": False,
            "is_forwarded": False,
            "source_file": "test.txt",
            "raw_line_start": 2,
            "raw_line_end": 2,
            "parse_warnings": [],
        },
    ]

    with open(input_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    report = clean_dataset(str(input_file), str(output_file), str(report_file))

    assert report["total_messages_processed"] == 2
    assert report["style_distributions"]["messages_with_emojis"] == 1
    assert report["style_distributions"]["messages_with_elongations"] == 1
    assert report["style_distributions"]["messages_with_all_caps"] == 1

    # Check output jsonl
    with open(output_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]

    assert len(lines) == 2
    assert lines[0]["sender"] == "You"
    assert lines[0]["views"]["analysis_text"] == "broo this is FUCKED 😭 😭"
    assert lines[1]["sender"] == "Friend_01"
    assert lines[1]["views"]["analysis_text"] == "Normal text here"
