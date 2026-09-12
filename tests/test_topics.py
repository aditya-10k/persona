"""
Unit tests for Stage T015: Topic Discovery & Tagging Engine.
Tests run on fast synthetic data (< 1 second) without loading transformer weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.nlp.topics import CONTENT_STOPWORDS, TopicConfig, TopicDiscoveryEngine, TopicModelResult


@pytest.fixture
def synthetic_topics_data() -> tuple[np.ndarray, list[dict], np.ndarray]:
    """
    Creates 3 synthetic topics in 16 dimensions with distinct vocabulary and known styles:
    Topic 0: Placements & Interviews (Style 1: Venting)
    Topic 1: APIs & Code Debugging (Style 4: Technical Collab)
    Topic 2: Gym & Workout (Style 2: Reactive Slang)
    """
    rng = np.random.RandomState(42)
    dim = 16
    n_per_topic = 25
    vectors = []
    pairs = []
    style_labels = []

    topic_configs = [
        # (topic_name, context_content, target_content, assigned_style)
        ("placement", "interview shortlisting resume hr test round", "bhai company ne shortlist nahi kia bc", 1),
        ("coding", "curl endpoint reservation api payload json schema", "thike me bolta hu backend check kar", 4),
        ("fitness", "gym workout benchpress sets protein deadlift", "aaj leg day tha bhai 💀", 2),
    ]

    for t_id, (topic_name, ctx, tgt, style_id) in enumerate(topic_configs):
        base_v = np.zeros(dim, dtype=np.float32)
        base_v[t_id] = 1.0  # Orthogonal basis vector

        for i in range(n_per_topic):
            noise = rng.normal(0, 0.08, size=dim).astype(np.float32)
            v = base_v + noise
            v = v / np.linalg.norm(v)
            vectors.append(v)

            pid = f"chat_{topic_name}:c000{t_id}:p{i:04d}"
            pairs.append({
                "pair_id": pid,
                "context_text": f"Friend: {ctx} batch_{i}",
                "context_analysis_text": f"{ctx} {i}",
                "target_text": f"{tgt} #{i}",
                "target_analysis_text": f"{tgt} {i}",
                "source_file": f"{topic_name}.txt",
            })
            style_labels.append(style_id)

    return np.array(vectors, dtype=np.float32), pairs, np.array(style_labels, dtype=int)


def test_content_stopwords_filter():
    """Verifies that conversational slang is filtered out while domain terms are retained."""
    engine = TopicDiscoveryEngine()
    stop_set = engine.config.content_stopwords

    # Slang and conversational markers must be in stop set
    assert "bhai" in stop_set
    assert "lol" in stop_set
    assert "bc" in stop_set
    assert "bkl" in stop_set
    assert "nahi" in stop_set
    assert "kuch" in stop_set
    assert "huh" in stop_set

    # Real topic domain terms must NOT be in stop set
    assert "placement" not in stop_set
    assert "interview" not in stop_set
    assert "api" not in stop_set
    assert "curl" not in stop_set
    assert "gym" not in stop_set
    assert "workout" not in stop_set


def test_topic_engine_fit(synthetic_topics_data):
    vecs, pairs, _ = synthetic_topics_data
    engine = TopicDiscoveryEngine(TopicConfig(random_seed=42))

    res = engine.fit(vecs, pairs, n_topics=3)

    assert isinstance(res, TopicModelResult)
    assert res.n_topics == 3
    assert len(res.labels) == len(vecs)
    assert res.centroids.shape == (3, 16)

    # Centroids must be unit normalized
    norms = np.linalg.norm(res.centroids, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-4)

    # Keywords must capture domain terms and exclude slang
    all_extracted_words = set()
    for prof in res.topic_profiles:
        words = [kw["term"] for kw in prof["keywords"]]
        all_extracted_words.update(words)
        # Ensure no banned conversational slang leaked into topic keywords
        for w in words:
            assert w not in CONTENT_STOPWORDS

    # Domain keywords should be discovered
    assert any("interview" in w or "resume" in w or "shortlisting" in w for w in all_extracted_words)
    assert any("payload" in w or "endpoint" in w or "reservation" in w for w in all_extracted_words)
    assert any("workout" in w or "benchpress" in w or "deadlift" in w for w in all_extracted_words)


def test_topic_evaluate_k(synthetic_topics_data):
    vecs, _, _ = synthetic_topics_data
    engine = TopicDiscoveryEngine(TopicConfig(random_seed=42))

    eval_res = engine.evaluate_k(vecs, k_range=[2, 3, 4])
    assert "k_evaluations" in eval_res
    assert "optimal_k" in eval_res
    assert eval_res["optimal_k"] in [2, 3, 4]
    assert eval_res["best_silhouette"] > 0.4


def test_compute_style_topic_matrix(synthetic_topics_data):
    vecs, pairs, style_labels = synthetic_topics_data
    engine = TopicDiscoveryEngine(TopicConfig(random_seed=42))

    res = engine.fit(vecs, pairs, n_topics=3)
    matrix_res = engine.compute_style_topic_matrix(res.labels, style_labels)

    probabilities = matrix_res["probabilities"]
    dominant = matrix_res["dominant_styles"]

    assert len(probabilities) == 3
    for topic_name, style_probs in probabilities.items():
        # Probabilities across styles for each topic must sum to 1.0
        total_p = sum(style_probs.values())
        assert pytest.approx(total_p, rel=1e-3) == 1.0

        # Each topic must have a dominant style
        assert topic_name in dominant
        assert dominant[topic_name]["probability"] > 0.5


def test_invalid_inputs():
    engine = TopicDiscoveryEngine()

    with pytest.raises(ValueError, match="must be 2D"):
        engine._ensure_normalized(np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="cannot be empty"):
        engine._ensure_normalized(np.zeros((0, 4)))

    with pytest.raises(ValueError, match="must match"):
        engine.fit(np.zeros((5, 4)), [{"pair_id": "1"}])
