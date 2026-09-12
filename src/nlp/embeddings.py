"""
Stage T012: Transformer Embedding Layer
Extracts contextual dense semantic representations using lightweight, CPU-efficient
Sentence-Transformer models (default: l3cube-pune/hinglish-sentence-bert).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "l3cube-pune/hindi-sentence-bert-nli"
FALLBACK_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class EmbeddingConfig:
    """Configuration for transformer embedding generation."""

    model_name: str = DEFAULT_MODEL
    batch_size: int = 64
    device: str = "cpu"
    normalize_embeddings: bool = True
    cache_dir: Optional[str] = None
    max_seq_length: int = 128


class EmbeddingEngine:
    """
    Sentence-Transformer embedding engine with batch processing,
    lazy model initialization, L2 normalization, and compressed persistence.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self._dimension: Optional[int] = None

    @property
    def model(self):
        """Lazy loader for SentenceTransformer to keep imports and init lightweight."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model '%s' on %s...", self.config.model_name, self.config.device)
            try:
                # Fast path: load directly from local cache without network checks
                self._model = SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device,
                    cache_folder=self.config.cache_dir,
                    local_files_only=True,
                )
            except Exception:
                # Fallback: fetch from Hugging Face if not yet cached
                self._model = SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device,
                    cache_folder=self.config.cache_dir,
                )
            if hasattr(self._model, "max_seq_length"):
                self._model.max_seq_length = self.config.max_seq_length
            # Cache dimension
            dummy = self._model.encode(["test"], show_progress_bar=False, normalize_embeddings=self.config.normalize_embeddings)
            self._dimension = int(dummy.shape[1])
            logger.info("Model loaded successfully. Embedding dimension: %d", self._dimension)
        return self._model

    @property
    def dimension(self) -> int:
        """Returns embedding dimension."""
        if self._dimension is None:
            _ = self.model  # triggers lazy load & dimension discovery
        return self._dimension

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
        normalize_embeddings: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Encode a text or sequence of texts into dense vectors.
        Returns float32 numpy array: [dimension] for single str, or [N, dimension] for sequence.
        """
        if isinstance(texts, str):
            single = True
            cleaned_texts = [texts]
        else:
            single = False
            if not texts:
                dim = self.dimension
                return np.empty((0, dim), dtype=np.float32)
            cleaned_texts = [str(t) if t is not None else "" for t in texts]

        bs = batch_size or self.config.batch_size
        norm = self.config.normalize_embeddings if normalize_embeddings is None else normalize_embeddings

        vectors = self.model.encode(
            cleaned_texts,
            batch_size=bs,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=norm,
            convert_to_numpy=True,
        )
        if single:
            return vectors[0].astype(np.float32)
        return vectors.astype(np.float32)

    def embed_batch(self, texts: Sequence[str], **kwargs) -> np.ndarray:
        """Alias for encode."""
        return self.encode(texts, **kwargs)

    def encode_pairs(
        self,
        pairs: Sequence[Dict[str, Any]],
        text_field: str = "target_analysis_text",
        fallback_field: str = "target_text",
        show_progress_bar: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract specified text field from dataset pairs and encode.
        Returns (vectors, pair_ids).
        """
        pair_ids = []
        texts_to_encode = []

        for pair in pairs:
            pair_id = pair.get("pair_id", "")
            pair_ids.append(pair_id)

            text = pair.get(text_field)
            if not text or not str(text).strip():
                text = pair.get(fallback_field, "")
            texts_to_encode.append(str(text or ""))

        vectors = self.encode(
            texts_to_encode,
            show_progress_bar=show_progress_bar,
        )
        return vectors, pair_ids

    def compute_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        If vectors are already L2-normalized, this is simply the dot product.
        """
        if vec_a.ndim == 1 and vec_b.ndim == 1:
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        raise ValueError("Inputs to compute_similarity must be 1D vectors.")

    def save_embeddings(
        self,
        output_dir: Path | str,
        name_prefix: str,
        vectors: np.ndarray,
        pair_ids: List[str],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, Path]:
        """
        Save embeddings to compressed .npz and accompanying metadata .json.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        npz_file = out_path / f"{name_prefix}_vectors.npz"
        meta_file = out_path / f"{name_prefix}_metadata.json"

        # Save vectors as compressed numpy archive
        np.savez_compressed(npz_file, vectors=vectors)

        metadata = {
            "embedding_model": self.config.model_name,
            "embedding_dimension": int(vectors.shape[1]) if vectors.ndim > 1 else 0,
            "total_vectors": int(vectors.shape[0]),
            "normalized": bool(self.config.normalize_embeddings),
            "pair_ids": pair_ids,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info("Saved %d embeddings to %s and %s", len(pair_ids), npz_file.name, meta_file.name)
        return npz_file, meta_file

    @staticmethod
    def load_embeddings(
        npz_file: Path | str,
        meta_file: Optional[Path | str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load embeddings from .npz and metadata .json.
        """
        npz_p = Path(npz_file)
        if not npz_p.exists():
            raise FileNotFoundError(f"Embedding file not found: {npz_p}")

        data = np.load(npz_p)
        vectors = data["vectors"]

        metadata = {}
        if meta_file:
            meta_p = Path(meta_file)
        else:
            meta_p = npz_p.with_name(npz_p.stem.replace("_vectors", "_metadata") + ".json")

        if meta_p.exists():
            with open(meta_p, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        return vectors, metadata
