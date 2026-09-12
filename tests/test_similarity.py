"""
Unit and integration tests for Stage T013: Semantic Similarity Engine
"""

import numpy as np
import pytest

from src.nlp.similarity import SimilarityEngine, SimilarityMatch


class DummyEmbeddingEngine:
    """Fast in-memory mock engine for unit testing similarity math without 950MB model load."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        # Returns deterministic unit vector pointing along axis 0
        arr = np.zeros((len(texts), self.dimension), dtype=np.float32)
        arr[:, 0] = 1.0
        return arr


@pytest.fixture
def sample_similarity_engine():
    """Creates a SimilarityEngine with 4 known synthetic normalized vectors."""
    # Orthogonal basis + one composite vector
    v0 = np.array([1.0, 0.0, 0.0, 0.0] + [0.0] * 764, dtype=np.float32)
    v1 = np.array([0.0, 1.0, 0.0, 0.0] + [0.0] * 764, dtype=np.float32)
    v2 = np.array([0.0, 0.0, 1.0, 0.0] + [0.0] * 764, dtype=np.float32)
    # 45-degree angle with v0 and v1: norm = 1.0
    v3 = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0] + [0.0] * 764, dtype=np.float32)

    vectors = np.vstack([v0, v1, v2, v3])
    pair_ids = ["pair_0", "pair_1", "pair_2", "pair_3"]
    pairs_data = [
        {"pair_id": "pair_0", "target_text": "text zero", "context_text": "context zero"},
        {"pair_id": "pair_1", "target_text": "text one", "context_text": "context one"},
        {"pair_id": "pair_2", "target_text": "text two", "context_text": "context two"},
        {"pair_id": "pair_3", "target_text": "text three", "context_text": "context three"},
    ]
    return SimilarityEngine(
        vectors=vectors,
        pair_ids=pair_ids,
        pairs_data=pairs_data,
        embedding_engine=DummyEmbeddingEngine(dimension=768),
    )


def test_find_similar_vector_ordering(sample_similarity_engine):
    """find_similar returns matches sorted descending by cosine similarity."""
    # Query identical to v0
    query = np.array([1.0, 0.0, 0.0, 0.0] + [0.0] * 764, dtype=np.float32)
    matches = sample_similarity_engine.find_similar(query=query, top_k=3)

    assert len(matches) == 3
    # Top match should be pair_0 with score 1.0
    assert matches[0].pair_id == "pair_0"
    assert matches[0].score == pytest.approx(1.0, abs=1e-4)
    assert matches[0].target_text == "text zero"

    # Second match should be pair_3 with score ~0.7071 (sqrt(0.5))
    assert matches[1].pair_id == "pair_3"
    assert matches[1].score == pytest.approx(np.sqrt(0.5), abs=1e-4)


def test_find_similar_min_score_filtering(sample_similarity_engine):
    """min_score filters out matches below threshold."""
    query = np.array([1.0, 0.0, 0.0, 0.0] + [0.0] * 764, dtype=np.float32)
    # Only v0 (1.0) and v3 (~0.707) should pass threshold 0.5
    matches = sample_similarity_engine.find_similar(query=query, top_k=10, min_score=0.5)

    assert len(matches) == 2
    assert {m.pair_id for m in matches} == {"pair_0", "pair_3"}


def test_pairwise_matrix_symmetry(sample_similarity_engine):
    """compute_pairwise_matrix produces symmetric matrix with unit diagonal."""
    gram = sample_similarity_engine.compute_pairwise_matrix()
    assert gram.shape == (4, 4)
    # Symmetry: M == M^T
    np.testing.assert_allclose(gram, gram.T, atol=1e-6)
    # Unit diagonal for L2-normalized vectors
    np.testing.assert_allclose(np.diag(gram), 1.0, atol=1e-6)


def test_compute_diversity_index(sample_similarity_engine):
    """compute_diversity_index computes valid stats within [0, 1]."""
    div_stats = sample_similarity_engine.compute_diversity_index()
    assert "mean_pairwise_similarity" in div_stats
    assert "diversity_index" in div_stats
    assert 0.0 <= div_stats["diversity_index"] <= 1.0
    assert div_stats["total_pairs_compared"] == 6  # 4*3/2


def test_context_response_alignment_breakdown():
    """compute_context_response_alignment correctly classifies mirroring vs reactive."""
    # 2 context-target pairs:
    # Pair 0: highly aligned (cosine = 1.0)
    # Pair 1: unaligned / reactive (cosine = 0.0)
    ctx = np.array([
        [1.0, 0.0, 0.0] + [0.0] * 765,
        [0.0, 1.0, 0.0] + [0.0] * 765,
    ], dtype=np.float32)

    tgt = np.array([
        [1.0, 0.0, 0.0] + [0.0] * 765,
        [0.0, 0.0, 1.0] + [0.0] * 765,
    ], dtype=np.float32)

    result = SimilarityEngine.compute_context_response_alignment(ctx, tgt)
    assert result["total_pairs"] == 2
    assert result["mean_alignment"] == pytest.approx(0.5, abs=1e-4)

    breakdown = result["behavior_breakdown"]
    assert breakdown["mirroring_count"] == 1  # score 1.0 >= 0.5
    assert breakdown["reactive_count"] == 1   # score 0.0 < 0.2
    assert breakdown["balanced_count"] == 0


def test_find_similar_with_text_query(sample_similarity_engine):
    """find_similar accepts string queries and returns SimilarityMatch objects."""
    matches = sample_similarity_engine.find_similar("hello test", top_k=2)
    assert len(matches) == 2
    assert all(isinstance(m, SimilarityMatch) for m in matches)
    # Dummy engine returns vector pointing along axis 0 -> pair_0 should be top match
    assert matches[0].pair_id == "pair_0"
    assert matches[0].score == pytest.approx(1.0, abs=1e-4)
