"""
Unit tests for Stage T014: Semantic Clustering Engine.
Tests run on fast synthetic data (< 1 second) without loading transformer weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.nlp.clustering import ClusterConfig, ClusteringEngine, ClusteringResult


@pytest.fixture
def synthetic_clusters() -> tuple[np.ndarray, list[str], list[dict]]:
    """
    Creates 3 well-separated synthetic clusters in 16 dimensions.
    Cluster 0: around +e0
    Cluster 1: around +e1
    Cluster 2: around +e2
    """
    rng = np.random.RandomState(42)
    dim = 16
    n_per_cluster = 25
    vectors = []
    pair_ids = []
    pairs_data = []

    cluster_topics = [
        ("code", "bhai bug fix kar diya repo me push kar diya", "tech"),
        ("humor", "haahaa bhai kya mast joke mara lol", "social"),
        ("gym", "aaj gym leg day hai workout done bro", "fitness"),
    ]

    for c_id, (topic, text, src) in enumerate(cluster_topics):
        base_v = np.zeros(dim, dtype=np.float32)
        base_v[c_id] = 1.0  # Orthogonal basis vector

        for i in range(n_per_cluster):
            noise = rng.normal(0, 0.1, size=dim).astype(np.float32)
            v = base_v + noise
            v = v / np.linalg.norm(v)
            vectors.append(v)

            pid = f"chat_{src}:c000{c_id}:p{i:04d}"
            pair_ids.append(pid)
            pairs_data.append({
                "pair_id": pid,
                "target_text": f"{text} #{i} 😂" if c_id == 1 else f"{text} #{i}",
                "target_analysis_text": f"{text} {i}",
                "context_text": f"Friend: what about {topic}?",
                "source_file": f"{src}.txt",
                "is_initiation": (i == 0),
                "sampling_strata": {"word_count": len(text.split()) + 1},
            })

    vecs = np.array(vectors, dtype=np.float32)
    return vecs, pair_ids, pairs_data


def test_ensure_normalized_vectors():
    engine = ClusteringEngine()

    # Test auto-normalization
    raw = np.array([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
    normalized = engine._ensure_normalized_vectors(raw)
    norms = np.linalg.norm(normalized, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)

    # Test invalid dimensions
    with pytest.raises(ValueError, match="must be 2D"):
        engine._ensure_normalized_vectors(np.array([1.0, 2.0]))

    # Test empty array
    with pytest.raises(ValueError, match="cannot be empty"):
        engine._ensure_normalized_vectors(np.zeros((0, 4)))


def test_evaluate_k(synthetic_clusters):
    vecs, _, _ = synthetic_clusters
    engine = ClusteringEngine(ClusterConfig(random_seed=42))

    eval_res = engine.evaluate_k(vecs, k_range=[2, 3, 4, 5])
    assert "k_evaluations" in eval_res
    assert "optimal_k" in eval_res
    assert 2 <= eval_res["optimal_k"] <= 5

    # Since there are 3 distinct orthogonal clusters, K=3 should produce high silhouette
    assert eval_res["best_silhouette"] > 0.4
    for k, metrics in eval_res["k_evaluations"].items():
        assert "silhouette" in metrics
        assert "calinski_harabasz" in metrics
        assert "inertia" in metrics


def test_fit_kmeans_determinism(synthetic_clusters):
    vecs, _, _ = synthetic_clusters
    engine1 = ClusteringEngine(ClusterConfig(random_seed=42))
    engine2 = ClusteringEngine(ClusterConfig(random_seed=42))

    res1 = engine1.fit(vecs, k=3)
    res2 = engine2.fit(vecs, k=3)

    assert np.array_equal(res1.labels, res2.labels)
    np.testing.assert_allclose(res1.centroids, res2.centroids, atol=1e-5)
    assert res1.silhouette == res2.silhouette


def test_centroids_unit_normalized(synthetic_clusters):
    vecs, _, _ = synthetic_clusters
    engine = ClusteringEngine()
    result = engine.fit(vecs, k=3)

    assert result.centroids.shape == (3, 16)
    norms = np.linalg.norm(result.centroids, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-4)


def test_extract_exemplars(synthetic_clusters):
    vecs, pair_ids, _ = synthetic_clusters
    engine = ClusteringEngine(ClusterConfig(exemplars_per_cluster=3))
    result = engine.fit(vecs, k=3)

    exemplars = engine.extract_exemplars(
        vecs, result.labels, pair_ids, top_n=3, centroids=result.centroids
    )

    assert len(exemplars) == 3
    for c_id, ex_list in exemplars.items():
        assert len(ex_list) == 3
        # Similarities must be descending
        sims = [ex["similarity"] for ex in ex_list]
        assert sims == sorted(sims, reverse=True)
        for ex in ex_list:
            assert 0.0 <= ex["similarity"] <= 1.01
            assert ex["pair_id"] in pair_ids


def test_profile_clusters(synthetic_clusters):
    vecs, pair_ids, pairs_data = synthetic_clusters
    engine = ClusteringEngine()
    result = engine.fit(vecs, k=3)

    exemplars = engine.extract_exemplars(vecs, result.labels, pair_ids, top_n=2)
    profiles = engine.profile_clusters(pairs_data, result.labels, exemplars=exemplars)

    assert len(profiles) == 3
    # Total percentage must sum to ~100%
    total_pct = sum(p["percentage"] for p in profiles)
    assert pytest.approx(total_pct, rel=1e-2) == 100.0

    # Test emoji signature in the humor cluster
    humor_profile = next(
        p for p in profiles if any(v["term"] == "haahaa" or v["term"] == "joke" for v in p["distinctive_vocabulary"])
    )
    assert len(humor_profile["emoji_signature"]) > 0
    assert humor_profile["emoji_signature"][0]["emoji"] == "😂"

    # Test distinctive vocabulary extraction (c-TF-IDF)
    for p in profiles:
        assert len(p["distinctive_vocabulary"]) > 0
        assert "target_word_stats" in p
        assert p["target_word_stats"]["mean"] > 0
        assert len(p["exemplars"]) == 2


def test_agglomerative_clustering(synthetic_clusters):
    vecs, _, _ = synthetic_clusters
    engine = ClusteringEngine(ClusterConfig(algorithm="agglomerative"))
    result = engine.fit(vecs, k=3)

    assert result.k == 3
    assert len(result.labels) == len(vecs)
    assert result.centroids.shape == (3, 16)
    norms = np.linalg.norm(result.centroids, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-4)
    assert result.silhouette > 0.4
