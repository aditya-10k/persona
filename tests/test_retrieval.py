"""
Unit tests for Stage T028 & T029: Style Embedding Index & Style Retrieval Engine.
Runs in < 1s on synthetic vectors without loading transformer weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.persona.retrieval import (
    RetrievalConfig,
    RetrievalItem,
    RetrievalResult,
    StyleEmbeddingIndex,
    StyleRetriever,
)


@pytest.fixture
def synthetic_index() -> StyleEmbeddingIndex:
    """Creates a synthetic 8-dimensional style index with 6 exemplars."""
    dim = 8
    # 6 exemplars across 3 situations:
    # 0, 1: technical_collab (short, medium)
    # 2, 3: casual_banter (short, long)
    # 4, 5: conflict_friction (medium, long)
    metadata = [
        {"example_id": "s001", "category": "technical_collab", "length_bin": "short", "context": "api error", "response": "curl check kar", "style_tags": ["all_lowercase"]},
        {"example_id": "s002", "category": "technical_collab", "length_bin": "medium", "context": "backend payload", "response": "json schema validate karna padega", "style_tags": ["hinglish_mixed"]},
        {"example_id": "s003", "category": "casual_banter", "length_bin": "short", "context": "kaha hai", "response": "ghar pe lol", "style_tags": ["all_lowercase", "slang_marker"]},
        {"example_id": "s004", "category": "casual_banter", "length_bin": "long", "context": "aaj kya plan he", "response": "kuch khas nahi bas chill karra hu and movie dekhunga", "style_tags": ["hinglish_mixed"]},
        {"example_id": "s005", "category": "conflict_friction", "length_bin": "medium", "context": "tune bola tha na", "response": "nahi bhai maine bilkul nahi bola", "style_tags": ["all_lowercase"]},
        {"example_id": "s006", "category": "conflict_friction", "length_bin": "long", "context": "code break hogaya", "response": "galat bolra he maine repo touch bhi nahi kia tha", "style_tags": ["hinglish_mixed"]},
    ]

    ctx_vecs = []
    resp_vecs = []

    # Assign distinct orthogonal bases per category
    basis = {
        "technical_collab": np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "casual_banter": np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        "conflict_friction": np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
    }

    rng = np.random.RandomState(42)
    for m in metadata:
        b = basis[m["category"]]
        c_v = b + rng.normal(0, 0.05, size=dim).astype(np.float32)
        r_v = b + rng.normal(0, 0.05, size=dim).astype(np.float32)
        ctx_vecs.append(c_v / np.linalg.norm(c_v))
        resp_vecs.append(r_v / np.linalg.norm(r_v))

    return StyleEmbeddingIndex(
        context_vectors=np.stack(ctx_vecs),
        response_vectors=np.stack(resp_vecs),
        metadata=metadata,
    )


def test_index_properties_and_normalization(synthetic_index):
    assert synthetic_index.size == 6
    assert synthetic_index.dimension == 8

    # Vectors must be unit-normalized
    ctx_norms = np.linalg.norm(synthetic_index.context_vectors, axis=1)
    np.testing.assert_allclose(ctx_norms, np.ones(6), atol=1e-5)

    resp_norms = np.linalg.norm(synthetic_index.response_vectors, axis=1)
    np.testing.assert_allclose(resp_norms, np.ones(6), atol=1e-5)

    # Hybrid vectors must also be normalized
    hybrid = synthetic_index.get_hybrid_vectors(alpha=0.5)
    np.testing.assert_allclose(np.linalg.norm(hybrid, axis=1), np.ones(6), atol=1e-5)


def test_index_save_and_load(synthetic_index, tmp_path):
    index_dir = tmp_path / "style_index"
    synthetic_index.save(index_dir)

    loaded = StyleEmbeddingIndex.load(index_dir)
    assert loaded.size == synthetic_index.size
    assert loaded.dimension == synthetic_index.dimension
    assert loaded.metadata == synthetic_index.metadata
    np.testing.assert_allclose(loaded.context_vectors, synthetic_index.context_vectors, atol=1e-5)
    np.testing.assert_allclose(loaded.response_vectors, synthetic_index.response_vectors, atol=1e-5)


def test_retriever_search_ordering(synthetic_index):
    retriever = StyleRetriever(synthetic_index, RetrievalConfig(top_k=2))

    # Query matching technical_collab basis (axis 0)
    q_tech = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    res = retriever.retrieve(query_vector=q_tech)

    assert res.retrieval_performed is True
    assert len(res.items) == 2
    # Top match must be a technical_collab exemplar
    assert res.items[0].example["category"] == "technical_collab"
    assert res.items[0].similarity_score > 0.90
    assert res.items[0].retrieval_rank == 1


def test_category_and_length_filtering(synthetic_index):
    retriever = StyleRetriever(synthetic_index, RetrievalConfig(top_k=3))
    q = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)

    # Filter for casual_banter only
    res_cat = retriever.retrieve(query_vector=q, category_filter="casual_banter")
    assert all(item.example["category"] == "casual_banter" for item in res_cat.items)

    # Filter for short length only
    res_len = retriever.retrieve(query_vector=q, length_filter="short")
    assert all(item.example["length_bin"] == "short" for item in res_len.items)


def test_selective_retrieval_logic(synthetic_index):
    retriever = StyleRetriever(synthetic_index, RetrievalConfig(selective_retrieval=True))

    # Trivial pings and greetings should NOT retrieve
    for ping in ["hi", "hey", "hello", "ha", "ok", "cool", "done", "bye", "hmm"]:
        needed, reason = retriever.should_retrieve(ping)
        assert needed is False, f"Expected no retrieval for: '{ping}'"
        assert "trivial" in reason.lower() or "greeting" in reason.lower()

    # Pure emoji / punctuation should NOT retrieve
    for emoji_ping in ["???", "...", "😭😭", "💀💀"]:
        needed, reason = retriever.should_retrieve(emoji_ping)
        assert needed is False, f"Expected no retrieval for: '{emoji_ping}'"

    # Substantive prompts MUST retrieve
    substantive = "bhai curl endpoint run nahi ho raha backend me check kar"
    needed, reason = retriever.should_retrieve(substantive)
    assert needed is True
    assert "substantive" in reason.lower()


def test_selective_retrieval_early_exit(synthetic_index):
    retriever = StyleRetriever(synthetic_index, RetrievalConfig(selective_retrieval=True))

    res = retriever.retrieve(query_text="hi")
    assert res.retrieval_performed is False
    assert len(res.items) == 0
    assert res.formatted_few_shot_prompt == ""
    assert res.latency_ms >= 0.0


def test_few_shot_prompt_formatting(synthetic_index):
    retriever = StyleRetriever(synthetic_index, RetrievalConfig(top_k=2))
    q = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    res = retriever.retrieve(query_vector=q)
    prompt = res.formatted_few_shot_prompt

    assert "### Authentic Persona Style Exemplars:" in prompt
    assert "[Example 1]" in prompt
    assert "Context:" in prompt
    assert "Persona Response:" in prompt
    assert "ghar pe lol" in prompt or "kuch khas" in prompt


def test_invalid_arguments():
    with pytest.raises(ValueError, match="must match"):
        StyleEmbeddingIndex(np.zeros((2, 4)), np.zeros((3, 4)), [{}])

    index = StyleEmbeddingIndex(np.ones((1, 4)), np.ones((1, 4)), [{}])
    retriever = StyleRetriever(index)

    with pytest.raises(ValueError, match="Either query_text or query_vector"):
        retriever.retrieve(query_text=None, query_vector=None)
