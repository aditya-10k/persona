"""
Stage T028 & T029: Style Embedding Index & Selective Style Retrieval Engine.
Provides high-speed local vector indexing of representative style exemplars
and context-aware, selective few-shot retrieval with MMR diversification.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Patterns for selective retrieval filtering
RE_TRIVIAL_PINGS = re.compile(
    r"^(hi|hey|hello|yo|sup|hlo|heyy|heyyy|ha|haa|haan|hnn|hn|ok|okay|k|kk|done|cool|nice|sahi|thik|theek|hmm|han|shi|yep|yes|bye|good night|gn|gm|good morning)[.!?\s]*$",
    re.IGNORECASE,
)
RE_ONLY_PUNCT_OR_EMOJI = re.compile(r"^[\s\W\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf]+$")


@dataclass
class RetrievalConfig:
    """Configuration for style exemplar retrieval."""

    top_k: int = 3
    retrieval_mode: str = "context"  # "context", "response", or "hybrid"
    hybrid_alpha: float = 0.6        # 60% context similarity, 40% response similarity
    diversity_lambda: float = 0.70   # MMR weight: 0.70 typicality vs 0.30 diversity
    min_similarity_threshold: float = 0.20  # Minimum cosine similarity to accept an exemplar
    selective_retrieval: bool = True # Skips retrieval on trivial pings/greetings
    min_words_for_retrieval: int = 2


@dataclass
class RetrievalItem:
    """Individual retrieved style exemplar with similarity score."""

    example: Dict[str, Any]
    similarity_score: float
    retrieval_rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example.get("example_id", ""),
            "category": self.example.get("category", ""),
            "cluster_label": self.example.get("cluster_label", ""),
            "response": self.example.get("response", ""),
            "context": self.example.get("context", ""),
            "style_tags": self.example.get("style_tags", []),
            "length_bin": self.example.get("length_bin", ""),
            "similarity_score": round(self.similarity_score, 4),
            "retrieval_rank": self.retrieval_rank,
        }


@dataclass
class RetrievalResult:
    """Complete output of a style retrieval query."""

    query_text: Optional[str]
    retrieval_performed: bool
    retrieval_reason: str
    items: List[RetrievalItem]
    formatted_few_shot_prompt: str
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_text": self.query_text,
            "retrieval_performed": self.retrieval_performed,
            "retrieval_reason": self.retrieval_reason,
            "items": [item.to_dict() for item in self.items],
            "formatted_few_shot_prompt": self.formatted_few_shot_prompt,
            "latency_ms": round(self.latency_ms, 3),
        }


class StyleEmbeddingIndex:
    """
    Local vector index storing dual (context & response) 768-dim embeddings
    and metadata for representative style exemplars.
    """

    def __init__(
        self,
        context_vectors: np.ndarray,
        response_vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
        embedding_model: str = "l3cube-pune/hindi-sentence-bert-nli",
    ):
        if len(context_vectors) != len(metadata) or len(response_vectors) != len(metadata):
            raise ValueError("Vectors and metadata lengths must match")

        self.context_vectors = self._normalize(context_vectors)
        self.response_vectors = self._normalize(response_vectors)
        self.metadata = metadata
        self.embedding_model = embedding_model

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """Ensures 2D unit-normalized float32 vectors."""
        vecs = vectors.astype(np.float32, copy=False)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vecs / norms

    @property
    def size(self) -> int:
        return len(self.metadata)

    @property
    def dimension(self) -> int:
        return self.context_vectors.shape[1] if self.size > 0 else 0

    def get_hybrid_vectors(self, alpha: float = 0.6) -> np.ndarray:
        """Computes combined normalized context + response vectors."""
        combined = alpha * self.context_vectors + (1.0 - alpha) * self.response_vectors
        return self._normalize(combined)

    def save(self, index_dir: Path) -> None:
        """Persists the vector index and metadata to disk."""
        index_dir.mkdir(parents=True, exist_ok=True)
        npz_path = index_dir / "style_index.npz"
        meta_path = index_dir / "style_index_metadata.json"

        np.savez_compressed(
            npz_path,
            context_vectors=self.context_vectors,
            response_vectors=self.response_vectors,
        )

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "size": self.size,
                "dimension": self.dimension,
                "embedding_model": self.embedding_model,
                "metadata": self.metadata,
            }, f, indent=2, ensure_ascii=False)

        logger.info("Saved StyleEmbeddingIndex (%d items, dim=%d) to %s",
                    self.size, self.dimension, index_dir)

    @classmethod
    def load(cls, index_dir: Path) -> StyleEmbeddingIndex:
        """Loads a persisted StyleEmbeddingIndex from disk."""
        npz_path = index_dir / "style_index.npz"
        meta_path = index_dir / "style_index_metadata.json"

        if not npz_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Index files not found at {index_dir}")

        with np.load(npz_path) as npz:
            ctx_vecs = npz["context_vectors"]
            resp_vecs = npz["response_vectors"]

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_doc = json.load(f)

        return cls(
            context_vectors=ctx_vecs,
            response_vectors=resp_vecs,
            metadata=meta_doc["metadata"],
            embedding_model=meta_doc.get("embedding_model", "l3cube-pune/hindi-sentence-bert-nli"),
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
        mode: str = "context",
        hybrid_alpha: float = 0.6,
        category_filter: Optional[str] = None,
        length_filter: Optional[str] = None,
        diversity_lambda: float = 0.70,
        min_similarity: float = 0.20,
    ) -> List[Tuple[int, float]]:
        """
        Executes MMR nearest-neighbor retrieval with optional category/length filtering.
        Returns list of (index, similarity_score).
        """
        if self.size == 0:
            return []

        # 1. Normalize query vector
        q = query_vector.astype(np.float32).reshape(1, -1)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        # 2. Select target vector space
        if mode == "response":
            target_matrix = self.response_vectors
        elif mode == "hybrid":
            target_matrix = self.get_hybrid_vectors(alpha=hybrid_alpha)
        else:
            target_matrix = self.context_vectors

        # 3. Filter candidates
        candidate_indices = []
        for i, item in enumerate(self.metadata):
            if category_filter and item.get("category") != category_filter:
                continue
            if length_filter and item.get("length_bin") != length_filter:
                continue
            candidate_indices.append(i)

        if not candidate_indices:
            # Fallback to all candidates if strict filter returned nothing
            candidate_indices = list(range(self.size))

        cand_vecs = target_matrix[candidate_indices]
        similarities = np.dot(cand_vecs, q.T).flatten()

        # Filter by minimum threshold
        valid_pairs = [
            (candidate_indices[idx], float(similarities[idx]))
            for idx in range(len(candidate_indices))
            if similarities[idx] >= min_similarity
        ]

        if not valid_pairs:
            # If nothing passes threshold, take top matches regardless
            best_order = np.argsort(-similarities)
            valid_pairs = [(candidate_indices[i], float(similarities[i])) for i in best_order[:top_k]]

        # 4. Maximal Marginal Relevance (MMR) for diversity
        cand_real_indices = [p[0] for p in valid_pairs]
        sim_to_query = {p[0]: p[1] for p in valid_pairs}

        selected: List[int] = []
        scores: List[float] = []

        # First match is the highest similarity to query
        first_match = max(cand_real_indices, key=lambda idx: sim_to_query[idx])
        selected.append(first_match)
        scores.append(sim_to_query[first_match])

        # Subsequent matches balance query sim with diversity
        cand_sub_matrix = target_matrix[cand_real_indices]
        idx_to_sub = {real_idx: i for i, real_idx in enumerate(cand_real_indices)}

        while len(selected) < min(top_k, len(cand_real_indices)):
            remaining = [idx for idx in cand_real_indices if idx not in selected]
            if not remaining:
                break

            rem_sub_indices = [idx_to_sub[idx] for idx in remaining]
            sel_sub_indices = [idx_to_sub[idx] for idx in selected]

            rem_vecs = cand_sub_matrix[rem_sub_indices]
            sel_vecs = cand_sub_matrix[sel_sub_indices]

            pw_sims = np.dot(rem_vecs, sel_vecs.T)
            max_sel_sim = np.max(pw_sims, axis=1)

            mmr_scores = []
            for i, real_idx in enumerate(remaining):
                sim_q = sim_to_query[real_idx]
                sim_d = max_sel_sim[i]
                score = diversity_lambda * sim_q - (1.0 - diversity_lambda) * sim_d
                mmr_scores.append(score)

            best_rem_idx = remaining[int(np.argmax(mmr_scores))]
            selected.append(best_rem_idx)
            scores.append(sim_to_query[best_rem_idx])

        return list(zip(selected, scores))


class StyleRetriever:
    """
    High-level selective style retrieval engine.
    Determines whether retrieval is necessary, queries the vector index,
    and formats few-shot prompts for downstream persona generation.
    """

    def __init__(
        self,
        index: StyleEmbeddingIndex,
        config: Optional[RetrievalConfig] = None,
        embedder: Optional[Any] = None,
    ):
        self.index = index
        self.config = config or RetrievalConfig()
        self.embedder = embedder

    def should_retrieve(self, query_text: str) -> Tuple[bool, str]:
        """
        Determines whether expensive few-shot retrieval is justified.
        Per T029: Trivial pings, simple greetings, or emoji-only inputs
        do not need retrieval because the static persona profile is sufficient.
        """
        if not self.config.selective_retrieval:
            return True, "Selective retrieval disabled; always retrieving."

        text = query_text.strip()
        if not text:
            return False, "Query is empty."

        # Check trivial greetings
        if RE_TRIVIAL_PINGS.match(text):
            return False, f"Static persona profile sufficient for trivial greeting/ping: '{text}'"

        # Check pure emoji or punctuation
        if RE_ONLY_PUNCT_OR_EMOJI.match(text):
            return False, "Static persona profile sufficient for pure emoji/punctuation."

        # Check length threshold
        words = text.split()
        if len(words) < self.config.min_words_for_retrieval:
            return False, f"Message length ({len(words)} words) below retrieval threshold."

        return True, "Context contains substantive conversational or situational content."

    def retrieve(
        self,
        query_text: Optional[str] = None,
        query_vector: Optional[np.ndarray] = None,
        category_filter: Optional[str] = None,
        length_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieves top relevant style exemplars given a query text or vector.
        """
        start_time = time.perf_counter()
        k = top_k or self.config.top_k

        # 1. Evaluate selective retrieval rule
        if query_text is not None:
            needed, reason = self.should_retrieve(query_text)
            if not needed:
                elapsed = (time.perf_counter() - start_time) * 1000.0
                return RetrievalResult(
                    query_text=query_text,
                    retrieval_performed=False,
                    retrieval_reason=reason,
                    items=[],
                    formatted_few_shot_prompt="",
                    latency_ms=elapsed,
                )
        else:
            reason = "Query vector supplied directly."

        # 2. Obtain query vector
        vec = query_vector
        if vec is None:
            if query_text is None:
                raise ValueError("Either query_text or query_vector must be provided")
            if self.embedder is None:
                raise ValueError("No embedder provided to encode query_text; supply query_vector or configure embedder.")
            vec = self.embedder.encode(query_text)

        # 3. Search vector index
        matches = self.index.search(
            query_vector=vec,
            top_k=k,
            mode=self.config.retrieval_mode,
            hybrid_alpha=self.config.hybrid_alpha,
            category_filter=category_filter,
            length_filter=length_filter,
            diversity_lambda=self.config.diversity_lambda,
            min_similarity=self.config.min_similarity_threshold,
        )

        items: List[RetrievalItem] = []
        for rank, (idx, sim) in enumerate(matches, 1):
            ex_meta = self.index.metadata[idx]
            items.append(RetrievalItem(
                example=ex_meta,
                similarity_score=sim,
                retrieval_rank=rank,
            ))

        # 4. Format few-shot prompt
        formatted_prompt = self.format_few_shot_prompt(items)
        elapsed = (time.perf_counter() - start_time) * 1000.0

        return RetrievalResult(
            query_text=query_text,
            retrieval_performed=True,
            retrieval_reason=reason,
            items=items,
            formatted_few_shot_prompt=formatted_prompt,
            latency_ms=elapsed,
        )

    @staticmethod
    def format_few_shot_prompt(items: List[RetrievalItem]) -> str:
        """
        Formats retrieved exemplars into an authentic few-shot in-context prompt block.
        """
        if not items:
            return ""

        blocks = ["### Authentic Persona Style Exemplars:"]
        for item in items:
            ex = item.example
            cat = ex.get("category", "general")
            tags = ", ".join(ex.get("style_tags", []))
            ctx = ex.get("context", "").strip()
            resp = ex.get("response", "").strip()

            block = (
                f"\n[Example {item.retrieval_rank}] ({cat} | tags: {tags})\n"
                f"Context:\n{ctx}\n"
                f"Persona Response:\n{resp}"
            )
            blocks.append(block)

        return "\n".join(blocks)
