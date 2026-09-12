"""
Stage T013: Semantic Similarity Engine
Provides sub-millisecond vectorized Top-K search, response diversity index,
context-response alignment analysis, and candidate fidelity scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from src.nlp.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


@dataclass
class SimilarityMatch:
    """Represents a single similarity search match."""

    pair_id: str
    score: float
    index: int
    target_text: str = ""
    context_text: str = ""
    source_file: str = ""
    is_initiation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "score": round(self.score, 4),
            "index": self.index,
            "target_text": self.target_text,
            "context_text": self.context_text,
            "source_file": self.source_file,
            "is_initiation": self.is_initiation,
        }


class SimilarityEngine:
    """
    High-performance semantic similarity engine operating on L2-normalized vectors.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        pair_ids: Sequence[str],
        pairs_data: Optional[Sequence[Dict[str, Any]]] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ):
        """
        Initialize the similarity engine.
        Vectors must be a 2D float32 numpy array [N, D].
        """
        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be 2D array, got shape {vectors.shape}")
        if len(pair_ids) != vectors.shape[0]:
            raise ValueError(
                f"Mismatch: {vectors.shape[0]} vectors but {len(pair_ids)} pair_ids"
            )

        self.vectors = vectors.astype(np.float32)
        self.pair_ids = list(pair_ids)
        self.embedding_engine = embedding_engine
        self.dimension = int(self.vectors.shape[1]) if self.vectors.shape[0] > 0 else 0
        self.total_vectors = int(self.vectors.shape[0])

        # Map pair_id to metadata if provided
        self.pairs_data = list(pairs_data) if pairs_data else []
        self._pair_lookup: Dict[str, Dict[str, Any]] = {}
        if self.pairs_data:
            for p in self.pairs_data:
                pid = p.get("pair_id")
                if pid:
                    self._pair_lookup[pid] = p

    def find_similar(
        self,
        query: Union[str, np.ndarray],
        top_k: int = 5,
        min_score: float = -1.0,
    ) -> List[SimilarityMatch]:
        """
        Find top-K most similar vectors to query.
        Query can be a raw text string (requires embedding_engine) or a 1D vector.
        """
        if self.total_vectors == 0:
            return []

        # 1. Resolve query vector
        if isinstance(query, str):
            if self.embedding_engine is None:
                raise ValueError("Cannot embed text query without an initialized EmbeddingEngine.")
            q_vec = self.embedding_engine.encode([query])[0]
        elif isinstance(query, np.ndarray):
            q_vec = query.astype(np.float32).flatten()
            if len(q_vec) != self.dimension:
                raise ValueError(
                    f"Query dimension {len(q_vec)} does not match engine dimension {self.dimension}"
                )
            # Ensure query is L2 normalized
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm
        else:
            raise TypeError(f"Query must be str or np.ndarray, got {type(query)}")

        # 2. Vectorized dot product against all N vectors: S = V . q
        scores = np.dot(self.vectors, q_vec)

        # 3. Filter by minimum score
        valid_indices = np.where(scores >= min_score)[0]
        if len(valid_indices) == 0:
            return []

        valid_scores = scores[valid_indices]

        # 4. Sort top_k descending
        k = min(top_k, len(valid_indices))
        if k < len(valid_indices):
            # Fast partial sort
            top_k_part = np.argpartition(-valid_scores, k - 1)[:k]
            top_indices = valid_indices[top_k_part]
            top_scores = scores[top_indices]
            # Exact sort of the top K items
            sort_order = np.argsort(-top_scores)
            sorted_indices = top_indices[sort_order]
            sorted_scores = top_scores[sort_order]
        else:
            sort_order = np.argsort(-valid_scores)
            sorted_indices = valid_indices[sort_order]
            sorted_scores = valid_scores[sort_order]

        # 5. Build results with enriched metadata
        results = []
        for idx, score in zip(sorted_indices, sorted_scores):
            pid = self.pair_ids[idx]
            meta = self._pair_lookup.get(pid, {})
            results.append(
                SimilarityMatch(
                    pair_id=pid,
                    score=float(score),
                    index=int(idx),
                    target_text=meta.get("target_text", ""),
                    context_text=meta.get("context_text", ""),
                    source_file=meta.get("source_file", ""),
                    is_initiation=bool(meta.get("is_initiation", False)),
                )
            )

        return results

    def compute_pairwise_matrix(self, subset_vectors: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute full Gram matrix M = V . V^T where M[i, j] = sim(v_i, v_j).
        """
        vecs = subset_vectors if subset_vectors is not None else self.vectors
        return np.dot(vecs, vecs.T)

    def compute_diversity_index(
        self,
        subset_vectors: Optional[np.ndarray] = None,
        sample_size: int = 1000,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Computes semantic diversity index across responses.
        Diversity Index = 1.0 - mean_pairwise_similarity.
        """
        vecs = subset_vectors if subset_vectors is not None else self.vectors
        n = len(vecs)
        if n < 2:
            return {
                "sample_size": n,
                "mean_pairwise_similarity": 1.0,
                "diversity_index": 0.0,
                "std_pairwise_similarity": 0.0,
            }

        # Subsample if large to prevent O(N^2) memory explosion
        if n > sample_size:
            rng = np.random.default_rng(random_seed)
            sampled_idx = rng.choice(n, size=sample_size, replace=False)
            sub_vecs = vecs[sampled_idx]
        else:
            sub_vecs = vecs

        # Compute upper triangle dot products (excluding diagonal i == j)
        gram = np.dot(sub_vecs, sub_vecs.T)
        triu_indices = np.triu_indices(len(sub_vecs), k=1)
        pairwise_sims = gram[triu_indices]

        mean_sim = float(np.mean(pairwise_sims))
        median_sim = float(np.median(pairwise_sims))
        std_sim = float(np.std(pairwise_sims))
        div_index = float(1.0 - mean_sim)

        return {
            "sample_size": len(sub_vecs),
            "total_pairs_compared": len(pairwise_sims),
            "mean_pairwise_similarity": round(mean_sim, 4),
            "median_pairwise_similarity": round(median_sim, 4),
            "std_pairwise_similarity": round(std_sim, 4),
            "min_pairwise_similarity": round(float(np.min(pairwise_sims)), 4),
            "max_pairwise_similarity": round(float(np.max(pairwise_sims)), 4),
            "diversity_index": round(div_index, 4),
        }

    @staticmethod
    def compute_context_response_alignment(
        context_vectors: np.ndarray,
        target_vectors: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calculates per-pair cosine similarity: align_i = context_i . target_i.
        Measures the extent to which the speaker mirrors the context vs pivots/reacts.
        """
        if len(context_vectors) != len(target_vectors):
            raise ValueError("Context and target vector counts must match.")
        if len(context_vectors) == 0:
            return {"count": 0}

        # Vectorized row-wise dot product: sum_j(C_ij * T_ij)
        alignments = np.sum(context_vectors * target_vectors, axis=1)

        mean_align = float(np.mean(alignments))
        median_align = float(np.median(alignments))
        std_align = float(np.std(alignments))

        # Categorize conversational behavior
        # High alignment (>= 0.50): Content mirroring / direct informational answer
        # Balanced (0.20 <= x < 0.50): Relevant conversational turn
        # Low (< 0.20): Reactive pivot, stylistic short reaction, or topic change
        mirroring_count = int(np.sum(alignments >= 0.50))
        balanced_count = int(np.sum((alignments >= 0.20) & (alignments < 0.50)))
        reactive_count = int(np.sum(alignments < 0.20))
        total = len(alignments)

        return {
            "total_pairs": total,
            "mean_alignment": round(mean_align, 4),
            "median_alignment": round(median_align, 4),
            "std_alignment": round(std_align, 4),
            "percentiles": {
                "p25": round(float(np.percentile(alignments, 25)), 4),
                "p50": round(float(np.percentile(alignments, 50)), 4),
                "p75": round(float(np.percentile(alignments, 75)), 4),
                "p90": round(float(np.percentile(alignments, 90)), 4),
            },
            "behavior_breakdown": {
                "mirroring_ratio": round(mirroring_count / total, 4),
                "mirroring_count": mirroring_count,
                "balanced_ratio": round(balanced_count / total, 4),
                "balanced_count": balanced_count,
                "reactive_ratio": round(reactive_count / total, 4),
                "reactive_count": reactive_count,
            },
        }

    def score_candidate(self, reference_text: str, candidate_text: str) -> float:
        """
        Compare a simulated/generated response against a human reference response.
        Used downstream in persona evaluation.
        """
        if self.embedding_engine is None:
            raise ValueError("score_candidate requires an EmbeddingEngine.")
        vecs = self.embedding_engine.encode([reference_text, candidate_text])
        return float(np.dot(vecs[0], vecs[1]))
