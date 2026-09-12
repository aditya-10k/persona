"""
Stage T036: Linguistic Similarity Evaluation.
Quantifies linguistic, stylistic, lexical, and semantic similarity between
generated candidate responses and authentic held-out target turns.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.nlp.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)

# Key signature persona markers to evaluate authentic vocabulary usage
SIGNATURE_VOCABULARY: Set[str] = {
    "bhai", "bro", "bc", "bkl", "tbh", "imo", "ngl", "waise", "bas", "are",
    "ha", "haa", "haan", "sahi", "shi", "thik", "theek", "cool", "done", "accha", "acha",
    "nahi", "nhi", "naa", "galat", "chodd", "scene", "fati", "legit", "fr", "bruh",
    "lol", "lmao", "ded", "api", "curl", "endpoint", "wsl", "json", "repo", "branch"
}

RE_EMOJI = re.compile(r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf]")
RE_TERMINAL_PERIOD = re.compile(r"\.\s*$")


@dataclass
class LinguisticMetricResult:
    """Quantitative linguistic and stylistic evaluation scores for a single response pair."""

    punctuation_score: float         # 1.0 = perfect adherence to unpunctuated rule; 0.0 = trailing period
    casing_score: float              # [0.0, 1.0] fidelity to persona casing conventions
    length_score: float              # [0.0, 1.0] 1.0 = exact length match; decreases with divergence
    lexical_jaccard: float           # [0.0, 1.0] word-level token overlap
    vocabulary_hit_rate: float       # [0.0, 1.0] presence of signature persona markers
    embedding_similarity: float      # [-1.0, 1.0] contextual embedding cosine similarity (768-dim)
    composite_score: float           # [0.0, 1.0] weighted composite linguistic fidelity score
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        for k in ["punctuation_score", "casing_score", "length_score", "lexical_jaccard",
                  "vocabulary_hit_rate", "embedding_similarity", "composite_score"]:
            res[k] = round(float(res[k]), 4)
        return res


class LinguisticEvaluator:
    """
    Evaluates candidate responses against authentic targets using surface,
    lexical, syntactic, and contextual transformer embedding metrics (T036).
    """

    def __init__(
        self,
        embedding_engine: Optional[EmbeddingEngine] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.embedding_engine = embedding_engine
        self.weights = weights or {
            "punctuation": 0.20,
            "casing": 0.15,
            "length": 0.15,
            "lexical": 0.15,
            "vocab_hit": 0.10,
            "embedding": 0.25,
        }
        # Normalize weights
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

    def evaluate_pair(
        self,
        candidate_text: str,
        target_text: str,
        target_context: str = "",
        candidate_vector: Optional[np.ndarray] = None,
        target_vector: Optional[np.ndarray] = None,
    ) -> LinguisticMetricResult:
        """
        Evaluate a single candidate response against the authentic target response.
        """
        cand = candidate_text.strip()
        targ = target_text.strip()

        # 1. Punctuation Score (Penalize trailing periods)
        # Target style: 92.5% unpunctuated. Having a trailing period is a severe stylistic error.
        has_trailing_period = bool(RE_TERMINAL_PERIOD.search(cand))
        target_has_period = bool(RE_TERMINAL_PERIOD.search(targ))
        if has_trailing_period and not target_has_period:
            punct_score = 0.0
        elif not has_trailing_period and not target_has_period:
            punct_score = 1.0
        elif has_trailing_period and target_has_period:
            punct_score = 0.9
        else:
            punct_score = 0.7  # target had period, cand didn't; acceptable since unpunctuated is persona default

        # 2. Casing Score
        # Persona default: ~43.5% lowercase, ~54.8% mixed case, ~1.7% caps.
        cand_is_lower = cand.islower()
        targ_is_lower = targ.islower()
        cand_is_upper = cand.isupper() and len(cand.split()) > 1
        targ_is_upper = targ.isupper() and len(targ.split()) > 1

        if cand_is_upper and not targ_is_upper:
            casing_score = 0.2  # Unwanted shouting
        elif cand_is_lower == targ_is_lower:
            casing_score = 1.0  # Perfect casing mode match
        elif cand_is_lower and not targ_is_lower:
            # Cand is lowercase, target is mixed case (acceptable casual variance)
            casing_score = 0.85
        else:
            # Cand is capitalized/sentence-cased when target is casual lowercase
            casing_score = 0.50

        # 3. Length Divergence Score
        cand_words = cand.split()
        targ_words = targ.split()
        len_c = max(1, len(cand_words))
        len_t = max(1, len(targ_words))

        # Scaled exponential penalty for length mismatch
        ratio = len_c / len_t
        if ratio >= 1.0:
            len_score = max(0.0, 1.0 - (ratio - 1.0) * 0.5)
        else:
            len_score = max(0.0, 1.0 - (1.0 / ratio - 1.0) * 0.4)
        len_score = min(1.0, max(0.0, len_score))

        # 4. Lexical Jaccard Overlap
        cand_tokens = set(re.findall(r"\b[a-zA-Z0-9_]+\b", cand.lower()))
        targ_tokens = set(re.findall(r"\b[a-zA-Z0-9_]+\b", targ.lower()))
        if not cand_tokens and not targ_tokens:
            jaccard = 1.0
        elif not cand_tokens or not targ_tokens:
            jaccard = 0.0
        else:
            intersection = cand_tokens.intersection(targ_tokens)
            union = cand_tokens.union(targ_tokens)
            jaccard = len(intersection) / len(union)

        # 5. Signature Vocabulary Hit Rate
        cand_hits = sum(1 for tok in cand_tokens if tok in SIGNATURE_VOCABULARY)
        targ_hits = sum(1 for tok in targ_tokens if tok in SIGNATURE_VOCABULARY)
        if targ_hits > 0:
            vocab_hit_rate = min(1.0, cand_hits / targ_hits)
        else:
            vocab_hit_rate = 1.0 if cand_hits > 0 or len_c <= 3 else 0.8

        # 6. Contextual Embedding Cosine Similarity
        if candidate_vector is not None and target_vector is not None:
            norm_c = np.linalg.norm(candidate_vector)
            norm_t = np.linalg.norm(target_vector)
            if norm_c > 1e-6 and norm_t > 1e-6:
                emb_sim = float(np.dot(candidate_vector, target_vector) / (norm_c * norm_t))
            else:
                emb_sim = 0.0
        elif self.embedding_engine is not None:
            vecs = self.embedding_engine.embed_batch([cand, targ])
            norm_c = np.linalg.norm(vecs[0])
            norm_t = np.linalg.norm(vecs[1])
            if norm_c > 1e-6 and norm_t > 1e-6:
                emb_sim = float(np.dot(vecs[0], vecs[1]) / (norm_c * norm_t))
            else:
                emb_sim = 0.0
        else:
            # Token character n-gram surrogate when embedding model is offline
            emb_sim = jaccard

        # Normalized embedding similarity mapped from [-1, 1] to [0, 1]
        norm_emb_sim = max(0.0, (emb_sim + 1.0) / 2.0) if emb_sim < 0 else emb_sim

        # 7. Composite Score
        composite = (
            self.weights["punctuation"] * punct_score
            + self.weights["casing"] * casing_score
            + self.weights["length"] * len_score
            + self.weights["lexical"] * jaccard
            + self.weights["vocab_hit"] * vocab_hit_rate
            + self.weights["embedding"] * norm_emb_sim
        )

        return LinguisticMetricResult(
            punctuation_score=punct_score,
            casing_score=casing_score,
            length_score=len_score,
            lexical_jaccard=jaccard,
            vocabulary_hit_rate=vocab_hit_rate,
            embedding_similarity=emb_sim,
            composite_score=min(1.0, max(0.0, composite)),
            details={
                "cand_words": len_c,
                "targ_words": len_t,
                "cand_is_lower": cand_is_lower,
                "targ_is_lower": targ_is_lower,
                "has_trailing_period": has_trailing_period,
                "target_has_period": target_has_period,
            },
        )

    def evaluate_batch(
        self,
        candidates: Sequence[str],
        targets: Sequence[str],
        contexts: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of candidate responses against targets.
        """
        if len(candidates) != len(targets):
            raise ValueError("Candidates and targets must have matching length.")

        n = len(candidates)
        if n == 0:
            return {
                "count": 0,
                "mean_composite_score": 0.0,
                "metrics": {},
                "results": [],
            }

        # Vectorized embeddings if engine provided
        cand_vecs: Optional[np.ndarray] = None
        targ_vecs: Optional[np.ndarray] = None
        if self.embedding_engine is not None:
            all_texts = list(candidates) + list(targets)
            all_vecs = self.embedding_engine.embed_batch(all_texts)
            cand_vecs = all_vecs[:n]
            targ_vecs = all_vecs[n:]

        results: List[LinguisticMetricResult] = []
        for i in range(n):
            c_vec = cand_vecs[i] if cand_vecs is not None else None
            t_vec = targ_vecs[i] if targ_vecs is not None else None
            ctx = contexts[i] if contexts and i < len(contexts) else ""
            res = self.evaluate_pair(
                candidates[i],
                targets[i],
                target_context=ctx,
                candidate_vector=c_vec,
                target_vector=t_vec,
            )
            results.append(res)

        # Aggregate summary statistics
        means = {
            "punctuation_score": float(np.mean([r.punctuation_score for r in results])),
            "casing_score": float(np.mean([r.casing_score for r in results])),
            "length_score": float(np.mean([r.length_score for r in results])),
            "lexical_jaccard": float(np.mean([r.lexical_jaccard for r in results])),
            "vocabulary_hit_rate": float(np.mean([r.vocabulary_hit_rate for r in results])),
            "embedding_similarity": float(np.mean([r.embedding_similarity for r in results])),
            "composite_score": float(np.mean([r.composite_score for r in results])),
        }

        return {
            "count": n,
            "mean_composite_score": round(means["composite_score"], 4),
            "means": {k: round(v, 4) for k, v in means.items()},
            "results": [r.to_dict() for r in results],
        }
