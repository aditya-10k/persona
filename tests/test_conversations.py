"""
Unit and integration tests for T007 Temporal Conversation Segmentation.
"""

import json
from pathlib import Path
import pytest

from src.segmentation.conversations import ConversationSegmenter, segment_dataset


@pytest.fixture
def segmenter():
    return ConversationSegmenter(gap_hours=4.0)


def test_time_gap_triggers_segmentation(segmenter):
    messages = [
        # Session 1: 10:00:00 to 10:05:00
        {
            "message_id": "file.txt:0001",
            "source_file": "file.txt",
            "timestamp": "2026-05-15T10:00:00",
            "sender": "Alice",
            "text": "Hello",
            "views": {"raw_text": "Hello", "normalized_text": "Hello"},
        },
        {
            "message_id": "file.txt:0002",
            "source_file": "file.txt",
            "timestamp": "2026-05-15T10:05:00",
            "sender": "You",
            "text": "Hi Alice",
            "views": {"raw_text": "Hi Alice", "normalized_text": "Hi Alice"},
        },
        # Session 2: 15:00:00 (5 hours later, > 4.0h gap)
        {
            "message_id": "file.txt:0003",
            "source_file": "file.txt",
            "timestamp": "2026-05-15T15:00:00",
            "sender": "Alice",
            "text": "Are you free now?",
            "views": {"raw_text": "Are you free now?", "normalized_text": "Are you free now?"},
        },
    ]

    sessions = list(segmenter.segment_messages(messages))

    assert len(sessions) == 2

    # Session 1
    s1 = sessions[0]
    assert s1["conversation_id"] == "file.txt:c0001"
    assert s1["message_count"] == 2
    assert s1["duration_seconds"] == 300
    assert s1["initiator"] == "Alice"
    assert s1["participants"] == ["Alice", "You"]
    assert s1["turn_count"] == 2

    # Session 2
    s2 = sessions[1]
    assert s2["conversation_id"] == "file.txt:c0002"
    assert s2["message_count"] == 1
    assert s2["duration_seconds"] == 0
    assert s2["initiator"] == "Alice"


def test_turn_grouping_burst_messages(segmenter):
    # When "You" sends 3 rapid messages, they must belong to ONE turn
    messages = [
        {
            "message_id": "f:1",
            "source_file": "f",
            "timestamp": "2026-05-15T10:00:00",
            "sender": "Bob",
            "text": "did you see the email?",
        },
        {
            "message_id": "f:2",
            "source_file": "f",
            "timestamp": "2026-05-15T10:01:00",
            "sender": "You",
            "text": "yes",
        },
        {
            "message_id": "f:3",
            "source_file": "f",
            "timestamp": "2026-05-15T10:01:10",
            "sender": "You",
            "text": "just replied",
        },
        {
            "message_id": "f:4",
            "source_file": "f",
            "timestamp": "2026-05-15T10:01:25",
            "sender": "You",
            "text": "check now",
        },
        {
            "message_id": "f:5",
            "source_file": "f",
            "timestamp": "2026-05-15T10:02:00",
            "sender": "Bob",
            "text": "got it!",
        },
    ]

    turns = segmenter.group_into_turns(messages)
    assert len(turns) == 3

    # Turn 1: Bob (1 msg)
    assert turns[0]["speaker"] == "Bob"
    assert turns[0]["message_count"] == 1

    # Turn 2: You (3 burst msgs)
    assert turns[1]["speaker"] == "You"
    assert turns[1]["message_count"] == 3
    assert turns[1]["message_ids"] == ["f:2", "f:3", "f:4"]

    # Turn 3: Bob (1 msg)
    assert turns[2]["speaker"] == "Bob"
    assert turns[2]["message_count"] == 1


def test_file_boundary_isolates_conversations(segmenter):
    messages = [
        {
            "message_id": "chatA.txt:0001",
            "source_file": "chatA.txt",
            "timestamp": "2026-05-15T10:00:00",
            "sender": "UserA",
            "text": "msg in A",
        },
        # Same timestamp, but different file!
        {
            "message_id": "chatB.txt:0001",
            "source_file": "chatB.txt",
            "timestamp": "2026-05-15T10:00:05",
            "sender": "UserB",
            "text": "msg in B",
        },
    ]

    sessions = list(segmenter.segment_messages(messages))
    assert len(sessions) == 2
    assert sessions[0]["source_file"] == "chatA.txt"
    assert sessions[0]["conversation_id"] == "chatA.txt:c0001"
    assert sessions[1]["source_file"] == "chatB.txt"
    assert sessions[1]["conversation_id"] == "chatB.txt:c0001"


def test_segment_dataset_integration(tmp_path):
    input_file = tmp_path / "cleaned_messages.jsonl"
    output_file = tmp_path / "conversations.jsonl"
    report_file = tmp_path / "segmentation_report.json"

    records = [
        {
            "message_id": "c1:1",
            "source_file": "c1.txt",
            "timestamp": "2026-05-15T10:00:00",
            "sender": "Alice",
            "text": "Hi",
            "views": {"raw_text": "Hi", "normalized_text": "Hi"},
        },
        {
            "message_id": "c1:2",
            "source_file": "c1.txt",
            "timestamp": "2026-05-15T10:02:00",
            "sender": "You",
            "text": "Hey Alice",
            "views": {"raw_text": "Hey Alice", "normalized_text": "Hey Alice"},
        },
        {
            "message_id": "c1:3",
            "source_file": "c1.txt",
            "timestamp": "2026-05-15T16:00:00",  # 6 hour gap
            "sender": "Alice",
            "text": "Second session",
            "views": {"raw_text": "Second session", "normalized_text": "Second session"},
        },
    ]

    with open(input_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    report = segment_dataset(str(input_file), str(output_file), str(report_file), gap_hours=4.0)

    assert report["total_conversations"] == 2
    assert report["total_messages_segmented"] == 3
    assert report["total_turns"] == 3
    assert report["conversation_size_distribution"]["1_msg"] == 1
    assert report["conversation_size_distribution"]["2_to_5_msgs"] == 1

    # Verify output jsonl
    with open(output_file, "r", encoding="utf-8") as f:
        convs = [json.loads(line) for line in f]

    assert len(convs) == 2
    assert convs[0]["conversation_id"] == "c1.txt:c0001"
    assert convs[0]["message_count"] == 2
    assert convs[1]["conversation_id"] == "c1.txt:c0002"
    assert convs[1]["message_count"] == 1
