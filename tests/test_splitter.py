"""
Unit and integration tests for T011 Train / Dev / Holdout Splitter.
"""

import json
from pathlib import Path
import pytest

from src.dataset.splitter import ConversationSplitter, create_dataset_splits


@pytest.fixture
def splitter():
    return ConversationSplitter(
        train_ratio=0.70, dev_ratio=0.15, test_ratio=0.15, random_seed=42
    )


def test_zero_leakage_and_valid_split_mapping(splitter):
    conversations = [
        {"conversation_id": f"src1:c{i:04d}", "source_file": "src1.txt"}
        for i in range(1, 21)
    ] + [
        {"conversation_id": f"src2:c{i:04d}", "source_file": "src2.txt"}
        for i in range(1, 11)
    ]

    mapping = splitter.build_split_mapping(conversations)

    # All conversations mapped
    assert len(mapping) == 30

    train_cids = {cid for cid, s in mapping.items() if s == "train"}
    dev_cids = {cid for cid, s in mapping.items() if s == "dev"}
    test_cids = {cid for cid, s in mapping.items() if s == "test"}

    # ZERO LEAKAGE
    assert len(train_cids & test_cids) == 0
    assert len(train_cids & dev_cids) == 0
    assert len(dev_cids & test_cids) == 0

    # Every split has data
    assert len(train_cids) > 0
    assert len(dev_cids) > 0
    assert len(test_cids) > 0


def test_seed_determinism():
    conversations = [
        {"conversation_id": f"chat:c{i:04d}", "source_file": "chat.txt"}
        for i in range(1, 30)
    ]

    s1 = ConversationSplitter(random_seed=99)
    s2 = ConversationSplitter(random_seed=99)

    m1 = s1.build_split_mapping(conversations)
    m2 = s2.build_split_mapping(conversations)

    assert m1 == m2


def test_split_pairs_assignment(splitter):
    split_mapping = {
        "c1": "train",
        "c2": "dev",
        "c3": "test",
    }

    pairs = [
        {"pair_id": "p1", "conversation_id": "c1", "target_text": "t1"},
        {"pair_id": "p2", "conversation_id": "c1", "target_text": "t2"},
        {"pair_id": "p3", "conversation_id": "c2", "target_text": "t3"},
        {"pair_id": "p4", "conversation_id": "c3", "target_text": "t4"},
    ]

    split_res = splitter.split_pairs(pairs, split_mapping)

    assert len(split_res["train"]) == 2
    assert len(split_res["dev"]) == 1
    assert len(split_res["test"]) == 1
    assert split_res["train"][0]["split"] == "train"
    assert split_res["test"][0]["split"] == "test"


def test_create_dataset_splits_integration(tmp_path):
    pairs_file = tmp_path / "sampled_pairs.jsonl"
    convs_file = tmp_path / "conversations.jsonl"
    out_dir = tmp_path / "splits"
    report_file = out_dir / "splits_report.json"

    convs = [
        {"conversation_id": f"s:c{i}", "source_file": "s.txt"}
        for i in range(1, 11)
    ]
    pairs = [
        {"pair_id": f"p{i}", "conversation_id": f"s:c{i}", "target_text": f"txt {i}", "source_file": "s.txt"}
        for i in range(1, 11)
    ]

    with open(convs_file, "w", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps(c) + "\n")

    with open(pairs_file, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    report = create_dataset_splits(
        str(pairs_file),
        str(convs_file),
        str(out_dir),
        str(report_file),
        train_ratio=0.70,
        dev_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )

    assert report["zero_leakage_verified"] is True
    assert report["conversations_split"]["total"] == 10
    assert report["pairs_split"]["total"] == 10

    # Check files exist
    assert (out_dir / "train_pairs.jsonl").exists()
    assert (out_dir / "dev_pairs.jsonl").exists()
    assert (out_dir / "test_pairs.jsonl").exists()
    assert (out_dir / "split_mapping.json").exists()
