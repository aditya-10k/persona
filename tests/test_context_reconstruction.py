"""
Unit and integration tests for T008 Context Reconstruction.
"""

import json
from pathlib import Path
import pytest

from src.segmentation.context_reconstruction import (
    ContextReconstructor,
    reconstruct_context_dataset,
)


@pytest.fixture
def reconstructor():
    return ContextReconstructor(max_context_turns=3, target_speaker="You")


def test_initiation_and_response_pairs(reconstructor):
    conversation = {
        "conversation_id": "chat.txt:c0001",
        "source_file": "chat.txt",
        "turns": [
            {
                "turn_id": 1,
                "speaker": "You",
                "start_timestamp": "2026-05-15T10:00:00",
                "end_timestamp": "2026-05-15T10:00:10",
                "message_count": 2,
                "message_ids": ["m1", "m2"],
            },
            {
                "turn_id": 2,
                "speaker": "Bob",
                "start_timestamp": "2026-05-15T10:01:00",
                "end_timestamp": "2026-05-15T10:01:05",
                "message_count": 1,
                "message_ids": ["m3"],
            },
            {
                "turn_id": 3,
                "speaker": "You",
                "start_timestamp": "2026-05-15T10:01:30",
                "end_timestamp": "2026-05-15T10:01:30",
                "message_count": 1,
                "message_ids": ["m4"],
            },
        ],
        "messages": [
            {
                "message_id": "m1",
                "sender": "You",
                "text": "Hey Bob",
                "views": {"raw_text": "Hey Bob", "analysis_text": "Hey Bob"},
            },
            {
                "message_id": "m2",
                "sender": "You",
                "text": "Are you there?",
                "views": {"raw_text": "Are you there?", "analysis_text": "Are you there?"},
            },
            {
                "message_id": "m3",
                "sender": "Bob",
                "text": "Yes I am here",
                "views": {"raw_text": "Yes I am here", "analysis_text": "Yes I am here"},
            },
            {
                "message_id": "m4",
                "sender": "You",
                "text": "Awesome",
                "views": {"raw_text": "Awesome", "analysis_text": "Awesome"},
            },
        ],
    }

    pairs = reconstructor.extract_pairs_from_conversation(conversation)
    assert len(pairs) == 2

    # Pair 1: Initiation
    p1 = pairs[0]
    assert p1["pair_id"] == "chat.txt:c0001:p0001"
    assert p1["is_initiation"] is True
    assert p1["context_depth"] == 0
    assert p1["context"] == []
    assert p1["response_latency_seconds"] is None
    assert p1["target_text"] == "Hey Bob\nAre you there?"
    assert p1["target_analysis_text"] == "Hey Bob Are you there?"

    # Pair 2: Reply to Bob
    p2 = pairs[1]
    assert p2["pair_id"] == "chat.txt:c0001:p0002"
    assert p2["is_initiation"] is False
    assert p2["context_depth"] == 2  # Window of 2 prior turns: Turn 1 (You) + Turn 2 (Bob)
    assert len(p2["context"]) == 2
    assert p2["context"][0]["speaker"] == "You"
    assert p2["context"][1]["speaker"] == "Bob"
    assert p2["target_text"] == "Awesome"
    # Latency: 10:01:05 (end of Bob's turn) to 10:01:30 (start of You's turn) = 25s
    assert p2["response_latency_seconds"] == 25.0


def test_max_context_window_limit(reconstructor):
    # 5 turns alternating, max_context_turns = 3
    turns = [
        {"turn_id": i, "speaker": "Bob" if i % 2 == 1 else "You", "start_timestamp": f"2026-05-15T10:0{i}:00", "end_timestamp": f"2026-05-15T10:0{i}:10", "message_count": 1, "message_ids": [f"m{i}"]}
        for i in range(1, 6)
    ]
    messages = [
        {"message_id": f"m{i}", "sender": "Bob" if i % 2 == 1 else "You", "text": f"turn {i}", "views": {"analysis_text": f"turn {i}"}}
        for i in range(1, 6)
    ]

    conv = {
        "conversation_id": "c:01",
        "source_file": "f.txt",
        "turns": turns,
        "messages": messages,
    }

    pairs = reconstructor.extract_pairs_from_conversation(conv)
    # Turns where speaker == "You": Turn 2 and Turn 4
    assert len(pairs) == 2

    # Turn 4 context: turns [1, 2, 3] (exactly 3 turns)
    p2 = pairs[1]
    assert p2["target_response"]["turn_id"] == 4
    assert p2["context_depth"] == 3
    assert [t["turn_id"] for t in p2["context"]] == [1, 2, 3]


def test_reconstruct_context_dataset_integration(tmp_path):
    input_file = tmp_path / "conversations.jsonl"
    output_file = tmp_path / "context_pairs.jsonl"
    report_file = tmp_path / "context_pairs_report.json"

    conv = {
        "conversation_id": "c1",
        "source_file": "c1.txt",
        "turns": [
            {"turn_id": 1, "speaker": "Alice", "start_timestamp": "2026-05-15T10:00:00", "end_timestamp": "2026-05-15T10:00:10", "message_count": 1, "message_ids": ["m1"]},
            {"turn_id": 2, "speaker": "You", "start_timestamp": "2026-05-15T10:00:20", "end_timestamp": "2026-05-15T10:00:25", "message_count": 1, "message_ids": ["m2"]},
        ],
        "messages": [
            {"message_id": "m1", "sender": "Alice", "text": "Hi", "views": {"analysis_text": "Hi"}},
            {"message_id": "m2", "sender": "You", "text": "Hello", "views": {"analysis_text": "Hello"}},
        ],
    }

    with open(input_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(conv) + "\n")

    report = reconstruct_context_dataset(
        str(input_file), str(output_file), str(report_file), max_context_turns=3, target_speaker="You"
    )

    assert report["total_context_pairs"] == 1
    assert report["reply_pairs"] == 1
    assert report["initiation_pairs"] == 0
    assert report["response_latency_stats"]["median_seconds"] == 10.0

    with open(output_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]

    assert len(lines) == 1
    assert lines[0]["target_text"] == "Hello"
    assert lines[0]["context_text"] == "Alice: Hi"
