"""
Tests for Stage T012: Transformer Embedding Layer
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.nlp.embeddings import EmbeddingConfig, EmbeddingEngine


@pytest.fixture(scope="module")
def shared_engine():
    """Module-scoped engine to avoid reloading model across tests."""
    config = EmbeddingConfig(model_name="l3cube-pune/hindi-sentence-bert-nli", batch_size=32)
    return EmbeddingEngine(config)


def test_embedding_config_defaults():
    """EmbeddingConfig provides sensible defaults."""
    cfg = EmbeddingConfig()
    assert cfg.model_name == "l3cube-pune/hindi-sentence-bert-nli"
    assert cfg.batch_size == 64
    assert cfg.device == "cpu"
    assert cfg.normalize_embeddings is True


def test_embedding_dimension(shared_engine):
    """l3cube-pune/hindi-sentence-bert-nli outputs 768 dimensions."""
    assert shared_engine.dimension == 768


def test_embedding_normalization(shared_engine):
    """Embeddings are strictly L2 normalized when normalize_embeddings=True."""
    texts = ["bhai sun na", "what are you doing tonight", "sahi hai bro"]
    vecs = shared_engine.encode(texts)
    assert vecs.shape == (3, 768)
    assert vecs.dtype == np.float32

    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_cosine_similarity_math(shared_engine):
    """compute_similarity computes exact dot product of normalized vectors."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    assert shared_engine.compute_similarity(v1, v2) == pytest.approx(1.0, abs=1e-6)
    assert shared_engine.compute_similarity(v1, v3) == pytest.approx(0.0, abs=1e-6)


def test_semantic_ranking_sanity(shared_engine):
    """Related conversational responses have higher similarity than unrelated topics."""
    t_anchor = "let's meet tomorrow evening"
    t_similar = "tomorrow evening sounds good, let's catch up"
    t_unrelated = "thermodynamics of black holes in astrophysics"

    vecs = shared_engine.encode([t_anchor, t_similar, t_unrelated])
    sim_similar = shared_engine.compute_similarity(vecs[0], vecs[1])
    sim_unrelated = shared_engine.compute_similarity(vecs[0], vecs[2])

    assert sim_similar > sim_unrelated
    assert sim_similar > 0.50
    assert sim_unrelated < 0.35


def test_encode_pairs(shared_engine):
    """encode_pairs extracts correct fields and returns pair_ids and vectors."""
    pairs = [
        {
            "pair_id": "test_chat:c001:p0001",
            "target_analysis_text": "chal done hai",
            "target_text": "chal done hai!!",
        },
        {
            "pair_id": "test_chat:c001:p0002",
            "target_analysis_text": "",
            "target_text": "fallback text",
        },
    ]
    vecs, ids = shared_engine.encode_pairs(pairs)
    assert len(ids) == 2
    assert ids[0] == "test_chat:c001:p0001"
    assert ids[1] == "test_chat:c001:p0002"
    assert vecs.shape == (2, 768)


def test_save_and_load_embeddings(shared_engine, tmp_path):
    """save_embeddings and load_embeddings preserve vectors and metadata."""
    texts = ["first test turn", "second test turn"]
    pair_ids = ["chat:c1:p1", "chat:c1:p2"]
    vecs = shared_engine.encode(texts)

    npz_file, meta_file = shared_engine.save_embeddings(
        output_dir=tmp_path,
        name_prefix="test_run",
        vectors=vecs,
        pair_ids=pair_ids,
        extra_metadata={"custom_flag": True},
    )

    assert npz_file.exists()
    assert meta_file.exists()

    loaded_vecs, loaded_meta = EmbeddingEngine.load_embeddings(npz_file, meta_file)
    np.testing.assert_array_almost_equal(vecs, loaded_vecs)
    assert loaded_meta["embedding_dimension"] == 768
    assert loaded_meta["total_vectors"] == 2
    assert loaded_meta["pair_ids"] == pair_ids
    assert loaded_meta["custom_flag"] is True


def test_empty_input(shared_engine):
    """Encoding empty list returns empty 2D array of shape (0, 768)."""
    vecs = shared_engine.encode([])
    assert vecs.shape == (0, 768)
