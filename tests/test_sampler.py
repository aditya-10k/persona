"""
Unit and integration tests for T010 Sampling Strategy.
"""

import json
from pathlib import Path
import pytest

from src.dataset.sampler import StratifiedSampler, sample_context_pairs_dataset


@pytest.fixture
def sampler():
    return StratifiedSampler(max_per_source=5, random_seed=42)


def test_classify_length_bin():
    assert StratifiedSampler.classify_length_bin("ha")[0] == "short"
    assert StratifiedSampler.classify_length_bin("bro that's crazy")[0] == "short"  # 3 words
    assert StratifiedSampler.classify_length_bin("yes I think we should do that")[0] == "medium"  # 7 words
    long_text = "this is a much longer message explaining how the entire pipeline is structured and executed across many different stages"
    assert StratifiedSampler.classify_length_bin(long_text)[0] == "long"  # 19 words


def test_undersized_source_100_percent_retained(sampler):
    # Only 3 pairs from a source, cap is 5 -> all 3 must be retained
    pairs = [
        {"pair_id": f"p{i}", "source_file": "small.txt", "target_text": f"text {i}", "is_initiation": False}
        for i in range(1, 4)
    ]
    sampled = sampler.sample_dataset(pairs)
    assert len(sampled) == 3
    assert [p["pair_id"] for p in sampled] == ["p1", "p2", "p3"]


def test_oversized_source_capped(sampler):
    # 20 pairs from a source, cap is 5 -> exactly 5 retained
    pairs = [
        {"pair_id": f"p{i}", "source_file": "large.txt", "target_text": f"message number {i}", "is_initiation": False}
        for i in range(1, 21)
    ]
    sampled = sampler.sample_dataset(pairs)
    assert len(sampled) == 5
    for p in sampled:
        assert p["source_file"] == "large.txt"
        assert "sampling_strata" in p


def test_initiations_100_percent_retained(sampler):
    # 2 initiations + 10 replies. Cap is 5.
    pairs = [
        {"pair_id": "init_1", "source_file": "f.txt", "target_text": "opening line 1", "is_initiation": True},
        {"pair_id": "init_2", "source_file": "f.txt", "target_text": "opening line 2", "is_initiation": True},
    ] + [
        {"pair_id": f"reply_{i}", "source_file": "f.txt", "target_text": f"reply {i}", "is_initiation": False}
        for i in range(1, 11)
    ]

    sampled = sampler.sample_dataset(pairs)
    assert len(sampled) == 5
    sampled_ids = [p["pair_id"] for p in sampled]
    # Both initiations must be preserved!
    assert "init_1" in sampled_ids
    assert "init_2" in sampled_ids


def test_seed_determinism():
    pairs = [
        {"pair_id": f"p{i}", "source_file": "f.txt", "target_text": f"word count {i} text", "is_initiation": False}
        for i in range(1, 50)
    ]
    sampler1 = StratifiedSampler(max_per_source=10, random_seed=123)
    sampler2 = StratifiedSampler(max_per_source=10, random_seed=123)

    sampled1 = sampler1.sample_dataset(pairs)
    sampled2 = sampler2.sample_dataset(pairs)

    assert [p["pair_id"] for p in sampled1] == [p["pair_id"] for p in sampled2]


def test_sample_dataset_integration(tmp_path):
    input_file = tmp_path / "filtered_pairs.jsonl"
    output_file = tmp_path / "sampled_pairs.jsonl"
    report_file = tmp_path / "sampling_report.json"

    pairs = [
        {"pair_id": f"p{i}", "source_file": "small.txt", "target_text": f"msg {i}", "is_initiation": False}
        for i in range(1, 4)
    ] + [
        {"pair_id": f"big_{i}", "source_file": "big.txt", "target_text": f"msg {i}", "is_initiation": False}
        for i in range(1, 20)
    ]

    with open(input_file, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    report = sample_context_pairs_dataset(
        str(input_file), str(output_file), str(report_file), max_per_source=5, random_seed=42
    )

    assert report["total_pairs_before"] == 22
    assert report["total_pairs_after"] == 8  # 3 from small + 5 from big
    assert report["source_distribution"]["small.txt"]["after"] == 3
    assert report["source_distribution"]["big.txt"]["after"] == 5

    with open(output_file, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]

    assert len(lines) == 8
