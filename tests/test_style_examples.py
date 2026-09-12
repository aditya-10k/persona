"""
Unit tests for Stage T026 & T027: Representative Style Dataset & Selection Engine.
Tests run in < 1s on synthetic data without loading transformer weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.persona.style_examples import (
    CLUSTER_NAMES,
    StyleExample,
    StyleExampleBuilder,
    StyleExampleConfig,
    StyleExampleResult,
    classify_length_bin,
    extract_style_tags,
)


def test_classify_length_bin():
    assert classify_length_bin(1) == "short"
    assert classify_length_bin(4) == "short"
    assert classify_length_bin(5) == "medium"
    assert classify_length_bin(14) == "medium"
    assert classify_length_bin(15) == "long"
    assert classify_length_bin(50) == "long"


def test_extract_style_tags():
    # Test lowercase unpunctuated Hinglish slang
    t1 = "ha bhai check kar liya maine"
    tags1 = extract_style_tags(t1)
    assert "all_lowercase" in tags1
    assert "unpunctuated" in tags1
    assert "slang_marker" in tags1
    assert "hinglish_mixed" in tags1

    # Test all caps question with exclamation
    t2 = "WHAT ARE YOU DOING?!"
    tags2 = extract_style_tags(t2)
    assert "all_caps" in tags2
    assert "question" in tags2
    assert "multiple_punct" in tags2
    assert "english_dominant" in tags2

    # Test emojis and elongation
    t3 = "broooo wait 😭😭"
    tags3 = extract_style_tags(t3)
    assert "elongation" in tags3
    assert "emoji_present" in tags3
    assert "emoji_burst" in tags3

    # Test verbless fragment
    t4 = "cool ok"
    tags4 = extract_style_tags(t4)
    assert "verbless_fragment" in tags4

    # Test multi bubble
    t5 = "first line\nsecond line"
    tags5 = extract_style_tags(t5, is_multi_bubble=True)
    assert "multi_bubble" in tags5


def test_mmr_selection_diversity():
    builder = StyleExampleBuilder()

    dim = 8
    # Centroid along axis 0
    centroid = np.zeros(dim, dtype=np.float32)
    centroid[0] = 1.0

    # Create 4 candidates:
    # 0 and 1 are almost identical to centroid
    # 2 is slightly different
    # 3 is orthogonal
    v0 = np.array([0.99, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    v0 = v0 / np.linalg.norm(v0)

    v1 = np.array([0.98, 0.11, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    v1 = v1 / np.linalg.norm(v1)

    v2 = np.array([0.8, 0.5, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    v2 = v2 / np.linalg.norm(v2)

    v3 = np.array([0.6, 0.7, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    v3 = v3 / np.linalg.norm(v3)

    vectors = np.stack([v0, v1, v2, v3])

    # Select 2 indices with MMR
    selected = builder._maximal_marginal_relevance(
        candidate_indices=[0, 1, 2, 3],
        vectors=vectors,
        centroid=centroid,
        k=2,
        lambda_param=0.5,
        max_sim=0.95,
    )

    assert len(selected) == 2
    # First must be 0 (closest to centroid)
    assert selected[0] == 0
    # Second should NOT be 1 because v1 is almost identical to v0 (> 0.95 sim)
    assert selected[1] in [2, 3]


def test_build_exemplars_synthetic():
    rng = np.random.RandomState(42)
    dim = 16
    pairs = []
    vectors = []
    cluster_map = {}

    situations_data = [
        ("technical_collab", "bhai curl endpoint payload json schema check kar", "thike me fix karta hu backend script", 4),
        ("casual_banter", "kaha hai bhai", "ghar pe hu chill karra hu lol", 1),
        ("conflict_friction", "tune commit kiya tha kya", "nahi maine kuch nahi kiya bhai galat mat bol", 0),
    ]

    idx = 0
    for sit_name, ctx, resp, cid in situations_data:
        base_v = np.zeros(dim, dtype=np.float32)
        base_v[cid] = 1.0

        for i in range(15):
            noise = rng.normal(0, 0.05, size=dim).astype(np.float32)
            v = base_v + noise
            v = v / np.linalg.norm(v)
            vectors.append(v)

            pid = f"chat_01.txt:c000{cid}:p{i:04d}"
            cluster_map[pid] = cid

            # Vary lengths: 5 short, 5 medium, 5 long
            if i < 5:
                r_text = f"ha sahi #{i}"
            elif i < 10:
                r_text = f"{resp} #{i}"
            else:
                r_text = f"{resp} and more extended text with explanations and details #{i}"

            pairs.append({
                "pair_id": pid,
                "conversation_id": f"chat_01.txt:c000{cid}",
                "source_file": "chat_01.txt",
                "context_text": f"{ctx} #{i}",
                "target_text": r_text,
            })
            idx += 1

    vectors_arr = np.array(vectors, dtype=np.float32)

    config = StyleExampleConfig(target_per_category=6, random_seed=42)
    builder = StyleExampleBuilder(config)

    res = builder.build_exemplars(
        pairs=pairs,
        vectors=vectors_arr,
        cluster_assignments=cluster_map,
    )

    assert isinstance(res, StyleExampleResult)
    assert res.total_examples > 0
    assert len(res.examples) == res.total_examples

    # Verify example schema
    ex = res.examples[0]
    assert isinstance(ex, StyleExample)
    assert ex.example_id.startswith("s")
    assert ex.category in ["technical_collab", "casual_banter", "conflict_friction", "acknowledgement", "career_academic", "advice_probing"]
    assert ex.cluster_id in [0, 1, 4]
    assert len(ex.style_tags) > 0
    assert ex.length_bin in ["short", "medium", "long"]
    assert ex.word_count > 0
    assert 0.0 <= ex.centroid_similarity <= 1.0
    assert ex.embedding_model == config.embedding_model

    # Verify serializability
    d = res.to_dict()
    assert "total_examples" in d
    assert "category_counts" in d
    assert "cluster_counts" in d
    assert "length_bin_counts" in d
    assert "mean_centroid_similarity" in d
    assert "mean_pairwise_similarity" in d
    assert len(d["examples"]) == res.total_examples


def test_anonymization_of_source_ids():
    builder = StyleExampleBuilder()
    source_map = {
        "RawSecretFile.txt": "chat_01.txt",
        "RawSecretFile.txt:c0001": "chat_01.txt:c0001",
    }

    pairs = [{
        "pair_id": "RawSecretFile.txt:c0001:p0001",
        "conversation_id": "RawSecretFile.txt:c0001",
        "source_file": "RawSecretFile.txt",
        "context_text": "hello there",
        "target_text": "hey what is up",
    }]
    vecs = np.ones((1, 8), dtype=np.float32)
    vecs = vecs / np.linalg.norm(vecs)

    res = builder.build_exemplars(pairs, vecs, source_map=source_map)
    assert res.total_examples == 1
    ex = res.examples[0]
    assert "RawSecretFile" not in ex.source_conversation
    assert "RawSecretFile" not in ex.pair_id
    assert ex.source_conversation.startswith("chat_01")
    assert ex.pair_id.startswith("chat_01")


def test_empty_or_invalid_inputs():
    builder = StyleExampleBuilder()

    with pytest.raises(ValueError, match="cannot be empty"):
        builder.build_exemplars([], np.zeros((0, 8)))

    with pytest.raises(ValueError, match="must match"):
        builder.build_exemplars([{"target_text": "a"}], np.zeros((2, 8)))
