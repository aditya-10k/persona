"""
Stage T014: Semantic Clustering Engine
Provides unsupervised Spherical K-Means clustering, silhouette sweep evaluation,
representative real medoid/exemplar extraction, and class-based TF-IDF (c-TF-IDF)
cluster profiling to uncover distinct texting archetypes.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import emoji
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

logger = logging.getLogger(__name__)

# Common universal stopwords and placeholders to ignore during c-TF-IDF profiling
DEFAULT_STOPWORDS: Set[str] = {
    # English functional stopwords
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "can", "could", "will", "would", "shall", "should", "may", "might",
    "must", "i", "me", "my", "myself", "we", "our", "ours", "you", "your",
    "yours", "he", "him", "his", "she", "her", "it", "its", "they", "them",
    "their", "theirs", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "so", "than", "too", "very", "just", "now", "if", "not",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    # Masked entities & metadata tokens
    "url", "num", "email", "phone", "name", "date", "time", "deleted",
    "message", "omitted",
    # Universal Hindi copulas & grammatical connectors (appear in all topics)
    "hai", "hain", "ke", "ka", "ki", "ko", "se", "me", "mein", "pe", "par",
    "bhi", "toh", "to", "ye", "yeh", "wo", "woh", "kya", "kyu", "kyun",
    "ho", "tha", "thi", "the", "kar", "karna", "raha", "rahe", "rahi", "h",
}


@dataclass
class ClusterConfig:
    """Configuration for semantic clustering."""

    algorithm: str = "kmeans"  # "kmeans" or "agglomerative"
    k_range: List[int] = field(default_factory=lambda: list(range(6, 19)))
    random_seed: int = 42
    n_init: int = 10
    max_iter: int = 300
    exemplars_per_cluster: int = 5


@dataclass
class ClusteringResult:
    """Results from a clustering run."""

    k: int
    labels: np.ndarray
    centroids: np.ndarray  # Shape [K, D], L2-normalized
    silhouette: float
    calinski_harabasz: float
    inertia: Optional[float] = None
    cluster_sizes: Dict[int, int] = field(default_factory=dict)
    evaluation_history: Optional[Dict[int, Dict[str, float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "silhouette": round(self.silhouette, 4),
            "calinski_harabasz": round(self.calinski_harabasz, 2),
            "inertia": round(self.inertia, 2) if self.inertia is not None else None,
            "cluster_sizes": {str(c): sz for c, sz in self.cluster_sizes.items()},
            "evaluation_history": self.evaluation_history,
            "metadata": self.metadata,
        }


class ClusteringEngine:
    """
    Semantic clustering engine for conversational vectors.
    Operates on L2-normalized vector space, making Euclidean K-Means mathematically
    equivalent to Spherical / Cosine K-Means.
    """

    def __init__(self, config: Optional[ClusterConfig] = None):
        self.config = config or ClusterConfig()

    def _ensure_normalized_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Ensures vectors are float32 and unit-normalized."""
        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be 2D array [N, D], got shape {vectors.shape}")
        if vectors.shape[0] == 0:
            raise ValueError("Vectors array cannot be empty")

        vecs = vectors.astype(np.float32, copy=False)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        # Check if vectors are already normalized within numerical precision
        if not np.allclose(norms, 1.0, atol=1e-3):
            norms = np.where(norms == 0, 1.0, norms)
            vecs = vecs / norms
        return vecs

    def evaluate_k(
        self,
        vectors: np.ndarray,
        k_range: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """
        Sweeps through candidate values of K to compute Silhouette Score
        and Calinski-Harabasz Index, finding the natural clustering resolution.
        """
        vecs = self._ensure_normalized_vectors(vectors)
        n_samples = vecs.shape[0]
        eval_range = list(k_range or self.config.k_range)

        # Filter out invalid K values
        eval_range = [k for k in eval_range if 2 <= k < n_samples]
        if not eval_range:
            raise ValueError(f"No valid K values in range for {n_samples} samples.")

        evaluations: Dict[int, Dict[str, float]] = {}

        for k in eval_range:
            km = KMeans(
                n_clusters=k,
                random_state=self.config.random_seed,
                n_init=self.config.n_init,
                max_iter=self.config.max_iter,
            )
            labels = km.fit_predict(vecs)

            # Compute Silhouette score using cosine distance
            # For n_samples <= 5000, exact cosine silhouette is fast
            sil = float(silhouette_score(vecs, labels, metric="cosine"))
            ch = float(calinski_harabasz_score(vecs, labels))
            inertia = float(km.inertia_)

            evaluations[k] = {
                "silhouette": round(sil, 4),
                "calinski_harabasz": round(ch, 2),
                "inertia": round(inertia, 2),
            }
            logger.debug(
                "K=%d -> Silhouette=%.4f, Calinski-Harabasz=%.2f, Inertia=%.2f",
                k, sil, ch, inertia
            )

        # Pick optimal K based on maximum silhouette score
        optimal_k = max(evaluations.keys(), key=lambda k: evaluations[k]["silhouette"])

        return {
            "k_evaluations": evaluations,
            "optimal_k": optimal_k,
            "best_silhouette": evaluations[optimal_k]["silhouette"],
            "best_calinski_harabasz": evaluations[optimal_k]["calinski_harabasz"],
        }

    def fit(
        self,
        vectors: np.ndarray,
        k: Optional[int] = None,
    ) -> ClusteringResult:
        """
        Fits the clustering model on L2-normalized vectors.
        If K is not provided, evaluate_k is used to determine the optimal K.
        """
        vecs = self._ensure_normalized_vectors(vectors)
        evaluation_history: Optional[Dict[int, Dict[str, float]]] = None

        if k is None:
            eval_res = self.evaluate_k(vecs)
            k = eval_res["optimal_k"]
            evaluation_history = eval_res["k_evaluations"]

        if k < 2 or k >= vecs.shape[0]:
            raise ValueError(f"Invalid K={k} for sample size {vecs.shape[0]}")

        if self.config.algorithm == "kmeans":
            km = KMeans(
                n_clusters=k,
                random_state=self.config.random_seed,
                n_init=self.config.n_init,
                max_iter=self.config.max_iter,
            )
            labels = km.fit_predict(vecs)
            raw_centroids = km.cluster_centers_.astype(np.float32)
            inertia = float(km.inertia_)
        elif self.config.algorithm == "agglomerative":
            agg = AgglomerativeClustering(
                n_clusters=k,
                metric="cosine",
                linkage="average",
            )
            labels = agg.fit_predict(vecs)
            inertia = None
            # Compute raw centroids as mean of member vectors
            raw_centroids = np.zeros((k, vecs.shape[1]), dtype=np.float32)
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    raw_centroids[c] = np.mean(vecs[mask], axis=0)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.config.algorithm}")

        # Re-project centroids onto the unit hypersphere: mu_k = mu_k / ||mu_k||_2
        norms = np.linalg.norm(raw_centroids, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized_centroids = raw_centroids / norms

        # Quality metrics
        sil = float(silhouette_score(vecs, labels, metric="cosine"))
        ch = float(calinski_harabasz_score(vecs, labels))

        # Cluster size distribution
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(c): int(cnt) for c, cnt in zip(unique, counts)}

        return ClusteringResult(
            k=k,
            labels=labels,
            centroids=normalized_centroids,
            silhouette=sil,
            calinski_harabasz=ch,
            inertia=inertia,
            cluster_sizes=cluster_sizes,
            evaluation_history=evaluation_history,
            metadata={
                "algorithm": self.config.algorithm,
                "n_samples": vecs.shape[0],
                "dimension": vecs.shape[1],
                "random_seed": self.config.random_seed,
            },
        )

    def extract_exemplars(
        self,
        vectors: np.ndarray,
        labels: np.ndarray,
        pair_ids: Sequence[str],
        top_n: Optional[int] = None,
        centroids: Optional[np.ndarray] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Finds the top N real medoid messages closest to each cluster centroid.
        m_k = argmax_{i in C_k} (v_i . mu_k)
        """
        vecs = self._ensure_normalized_vectors(vectors)
        if len(labels) != len(vecs) or len(pair_ids) != len(vecs):
            raise ValueError("Lengths of vectors, labels, and pair_ids must match.")

        n_top = top_n or self.config.exemplars_per_cluster
        k = len(np.unique(labels))

        if centroids is None:
            # Compute centroids if not provided
            centroids = np.zeros((k, vecs.shape[1]), dtype=np.float32)
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    mean_v = np.mean(vecs[mask], axis=0)
                    norm = np.linalg.norm(mean_v)
                    centroids[c] = mean_v / (norm if norm > 0 else 1.0)

        exemplars: Dict[int, List[Dict[str, Any]]] = {}

        for c in range(k):
            cluster_indices = np.where(labels == c)[0]
            if len(cluster_indices) == 0:
                exemplars[c] = []
                continue

            cluster_vectors = vecs[cluster_indices]
            # Cosine similarity is dot product of normalized vectors
            similarities = np.dot(cluster_vectors, centroids[c])
            sorted_order = np.argsort(-similarities)

            top_indices = cluster_indices[sorted_order[:n_top]]
            top_sims = similarities[sorted_order[:n_top]]

            exemplars[c] = [
                {
                    "pair_id": pair_ids[idx],
                    "similarity": round(float(sim), 4),
                    "index": int(idx),
                }
                for idx, sim in zip(top_indices, top_sims)
            ]

        return exemplars

    def profile_clusters(
        self,
        pairs: Sequence[Dict[str, Any]],
        labels: np.ndarray,
        exemplars: Optional[Dict[int, List[Dict[str, Any]]]] = None,
        top_vocab_n: int = 10,
        top_emojis_n: int = 5,
        stopwords: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Computes rich cluster profiles including:
        - Cluster size & percentage of dataset
        - Word and character count distributions
        - Distinctive vocabulary using class-based TF-IDF (c-TF-IDF)
        - Emoji usage signatures
        - Initiations and chat source distributions
        - Real medoid exemplars with original texts
        """
        if len(pairs) != len(labels):
            raise ValueError("Length of pairs and labels must match.")

        stop_set = stopwords or DEFAULT_STOPWORDS
        total_samples = len(pairs)
        k = len(np.unique(labels))

        # Map pair_id -> pair dict for fast exemplar lookup
        pairs_by_id = {p.get("pair_id", ""): p for p in pairs}

        # Step 1: Collect tokens and emojis per cluster
        token_pattern = re.compile(r"[a-zA-Z\u0900-\u097F]{2,}")
        cluster_words: Dict[int, List[str]] = {c: [] for c in range(k)}
        cluster_emojis: Dict[int, List[str]] = {c: [] for c in range(k)}
        cluster_indices: Dict[int, List[int]] = {c: [] for c in range(k)}

        for idx, (pair, cluster_id) in enumerate(zip(pairs, labels)):
            c = int(cluster_id)
            cluster_indices[c].append(idx)

            target_text = pair.get("target_text", "")
            analysis_text = pair.get("target_analysis_text") or pair.get("target_text", "")

            # Extract emojis
            em_list = emoji.emoji_list(target_text)
            for em in em_list:
                cluster_emojis[c].append(em["emoji"])

            # Extract tokens for vocabulary profiling
            words = token_pattern.findall(analysis_text.lower())
            filtered_words = [w for w in words if w not in stop_set]
            cluster_words[c].extend(filtered_words)

        # Step 2: Calculate Document Frequency (DF) across clusters for c-TF-IDF
        # DF = number of clusters in which word w appears
        word_cluster_presence: Dict[str, Set[int]] = {}
        for c in range(k):
            unique_words = set(cluster_words[c])
            for w in unique_words:
                if w not in word_cluster_presence:
                    word_cluster_presence[w] = set()
                word_cluster_presence[w].add(c)

        # Step 3: Build detailed profile for each cluster
        profiles: List[Dict[str, Any]] = []

        for c in range(k):
            c_indices = cluster_indices[c]
            c_pairs = [pairs[i] for i in c_indices]
            size = len(c_pairs)
            percentage = round((size / total_samples) * 100, 2) if total_samples > 0 else 0.0

            # Target word and character counts
            target_word_counts = [
                p.get("sampling_strata", {}).get("word_count")
                or len(p.get("target_text", "").split())
                for p in c_pairs
            ]
            target_char_counts = [len(p.get("target_text", "")) for p in c_pairs]

            mean_target_words = round(float(np.mean(target_word_counts)), 2) if target_word_counts else 0.0
            median_target_words = round(float(np.median(target_word_counts)), 2) if target_word_counts else 0.0
            min_target_words = int(np.min(target_word_counts)) if target_word_counts else 0
            max_target_words = int(np.max(target_word_counts)) if target_word_counts else 0
            mean_target_chars = round(float(np.mean(target_char_counts)), 2) if target_char_counts else 0.0

            # Initiation ratio
            initiations = sum(1 for p in c_pairs if p.get("is_initiation", False))
            initiation_ratio = round(initiations / size, 4) if size > 0 else 0.0

            # Source file distribution
            source_counter = Counter(p.get("source_file", "unknown") for p in c_pairs)
            source_dist = {src: cnt for src, cnt in source_counter.most_common(5)}

            # Emoji signature
            emoji_counter = Counter(cluster_emojis[c])
            emoji_signature = [
                {"emoji": em, "count": cnt, "ratio": round(cnt / size, 4)}
                for em, cnt in emoji_counter.most_common(top_emojis_n)
            ]

            # c-TF-IDF Distinctive Vocabulary
            c_tf = Counter(cluster_words[c])
            total_cluster_words = len(cluster_words[c])
            c_tfidf_scores: Dict[str, float] = {}

            if total_cluster_words > 0:
                for word, count in c_tf.items():
                    # Only consider words with at least 2 occurrences in this cluster to avoid single typos
                    if count < 2 and size > 20:
                        continue
                    tf = count / total_cluster_words
                    df = len(word_cluster_presence.get(word, set()))
                    idf = math.log(1.0 + (k + 1.0) / (df + 1.0)) + 1.0
                    c_tfidf_scores[word] = tf * idf

            top_vocab = [
                {"term": w, "score": round(score, 6), "count": c_tf[w]}
                for w, score in sorted(c_tfidf_scores.items(), key=lambda x: -x[1])[:top_vocab_n]
            ]

            # Attach exemplar details
            cluster_exemplars: List[Dict[str, Any]] = []
            if exemplars and c in exemplars:
                for ex in exemplars[c]:
                    pid = ex.get("pair_id", "")
                    matched_pair = pairs_by_id.get(pid, {})
                    cluster_exemplars.append({
                        "pair_id": pid,
                        "similarity": ex.get("similarity", 0.0),
                        "target_text": matched_pair.get("target_text", ""),
                        "context_text": matched_pair.get("context_text", ""),
                        "source_file": matched_pair.get("source_file", ""),
                    })

            profiles.append({
                "cluster_id": c,
                "size": size,
                "percentage": percentage,
                "target_word_stats": {
                    "mean": mean_target_words,
                    "median": median_target_words,
                    "min": min_target_words,
                    "max": max_target_words,
                    "mean_chars": mean_target_chars,
                },
                "initiation_ratio": initiation_ratio,
                "source_distribution": source_dist,
                "emoji_signature": emoji_signature,
                "distinctive_vocabulary": top_vocab,
                "exemplars": cluster_exemplars,
            })

        # Sort profiles by cluster size descending for immediate readability
        profiles.sort(key=lambda p: -p["size"])
        return profiles
