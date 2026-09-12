"""
Stage T015: Topic Discovery & Tagging Engine
Discovers and tags conversational topics ('what is being discussed') independently
from communication style archetypes ('how it is spoken').
Computes the P(Style | Topic) Situational Transition Matrix for dynamic persona adaptation.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

logger = logging.getLogger(__name__)

# Aggressive conversational and functional stopwords to ensure pure topical content extraction
CONTENT_STOPWORDS: Set[str] = {
    # English functional words
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "can", "could", "will", "would", "shall", "should", "may", "might",
    "must", "i", "me", "my", "myself", "we", "our", "ours", "you", "your",
    "yours", "he", "him", "his", "she", "her", "it", "its", "they", "them",
    "their", "theirs", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "so", "than", "too", "very", "just", "now", "if", "not",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "there", "here", "when", "where", "why", "how", "then", "also", "out",
    "back", "only", "even", "such", "no", "yes",
    # Masked entities & system artifacts
    "url", "num", "email", "phone", "name", "date", "time", "deleted",
    "message", "omitted", "image", "audio", "video", "sticker", "contact",
    "vcard", "version", "begin", "end", "forwarded", "fn", "tel", "waid",
    # Universal Hindi copulas & grammatical connectors
    "hai", "hain", "ke", "ka", "ki", "ko", "se", "me", "mein", "pe", "par",
    "bhi", "toh", "to", "ye", "yeh", "wo", "woh", "kya", "kyu", "kyun",
    "ho", "tha", "thi", "the", "kar", "karna", "raha", "rahe", "rahi", "h",
    # Conversational slang & stylistic fillers (MUST be excluded from topics)
    "bhai", "yaar", "bro", "bc", "bkl", "lol", "lmao", "rofl", "haahaa", "haha",
    "hahaha", "nahi", "nhi", "na", "haan", "haa", "ha", "acha", "arre", "are",
    "abe", "abey", "btao", "bata", "bol", "bola", "bolra", "dekh", "dekha",
    "de", "diya", "die", "gaya", "gye", "gayi", "kuch", "aisa", "aise",
    "kese", "kaise", "thoda", "bahut", "bohot", "legit", "damn", "fr", "huh",
    "hmm", "hm", "oo", "oh", "ok", "okay", "nice", "bruh", "whi", "wahi",
    "fir", "phir", "pehle", "chodd", "mat", "kr", "krra", "mene", "maine",
    "tune", "tera", "teri", "tere", "mera", "meri", "mere", "apna", "apne",
    "apni", "uske", "uski", "uska", "inke", "inka", "inki", "unka", "unki",
    "unke", "sab", "sabko", "kisi", "koi", "yehi", "wohi", "log", "baat",
    "waala", "wala", "wali", "wale", "karo", "kare", "karenge", "karega",
    "karegi", "hoga", "hogi", "honge", "hota", "hoti", "hote", "hua", "hue",
    "hui", "pata", "lag", "rha", "rhe", "rhi", "chal", "chalo", "chalra",
    "mujhe", "tujhe", "hume", "unhe", "usse", "isse", "jaise", "wese",
    "mai", "aur", "mei", "nai", "hee", "vhi", "voh", "achaa", "ohh", "nah",
    "abhi", "wha", "waha", "yha", "yaha", "okk", "ohkk", "yess", "accha",
    "acchaa", "kia", "kiya", "lie", "liye", "joh", "bas", "karra", "karne",
    # Generic conversational roles & meta tokens (must not be topics)
    "you", "friend", "user", "admin", "system", "group",
}



@dataclass
class TopicConfig:
    """Configuration for topic discovery."""

    n_topics: Optional[int] = None
    k_range: List[int] = field(default_factory=lambda: list(range(5, 13)))
    random_seed: int = 42
    n_init: int = 10
    max_iter: int = 300
    top_keywords_n: int = 10
    exemplars_per_topic: int = 4
    content_stopwords: Set[str] = field(default_factory=lambda: set(CONTENT_STOPWORDS))


@dataclass
class TopicModelResult:
    """Results of conversational topic modeling."""

    n_topics: int
    labels: np.ndarray
    centroids: np.ndarray
    silhouette: float
    calinski_harabasz: float
    topic_profiles: List[Dict[str, Any]]
    evaluation_history: Optional[Dict[int, Dict[str, float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_topics": self.n_topics,
            "silhouette": round(self.silhouette, 4),
            "calinski_harabasz": round(self.calinski_harabasz, 2),
            "topic_profiles": self.topic_profiles,
            "evaluation_history": self.evaluation_history,
            "metadata": self.metadata,
        }


class TopicDiscoveryEngine:
    """
    Topic discovery engine that identifies domain themes from conversational embeddings
    and computes the P(Style | Topic) situational style transition probabilities.
    """

    def __init__(self, config: Optional[TopicConfig] = None):
        self.config = config or TopicConfig()

    def _ensure_normalized(self, vectors: np.ndarray) -> np.ndarray:
        """Validates and unit-normalizes vectors."""
        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be 2D, got shape {vectors.shape}")
        if vectors.shape[0] == 0:
            raise ValueError("Vectors array cannot be empty")

        vecs = vectors.astype(np.float32, copy=False)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        if not np.allclose(norms, 1.0, atol=1e-3):
            norms = np.where(norms == 0, 1.0, norms)
            vecs = vecs / norms
        return vecs

    def evaluate_k(
        self,
        vectors: np.ndarray,
        k_range: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Sweeps topic candidate count to evaluate clustering separation."""
        vecs = self._ensure_normalized(vectors)
        eval_range = list(k_range or self.config.k_range)
        eval_range = [k for k in eval_range if 2 <= k < len(vecs)]

        evaluations: Dict[int, Dict[str, float]] = {}
        for k in eval_range:
            km = KMeans(
                n_clusters=k,
                random_state=self.config.random_seed,
                n_init=self.config.n_init,
                max_iter=self.config.max_iter,
            )
            labels = km.fit_predict(vecs)
            sil = float(silhouette_score(vecs, labels, metric="cosine"))
            ch = float(calinski_harabasz_score(vecs, labels))
            inertia = float(km.inertia_)
            evaluations[k] = {
                "silhouette": round(sil, 4),
                "calinski_harabasz": round(ch, 2),
                "inertia": round(inertia, 2),
            }

        optimal_k = max(evaluations.keys(), key=lambda k: evaluations[k]["silhouette"])
        return {
            "k_evaluations": evaluations,
            "optimal_k": optimal_k,
            "best_silhouette": evaluations[optimal_k]["silhouette"],
        }

    def extract_topic_keywords(
        self,
        pairs: Sequence[Dict[str, Any]],
        labels: np.ndarray,
        top_n: Optional[int] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Extracts content-bearing keywords per topic using class-based TF-IDF (c-TF-IDF),
        strictly filtering out conversational slang and stylistic markers.
        """
        n_words = top_n or self.config.top_keywords_n
        stop_set = self.config.content_stopwords
        k = len(np.unique(labels))

        token_pattern = re.compile(r"[a-zA-Z\u0900-\u097F]{2,}")
        topic_words: Dict[int, List[str]] = {t: [] for t in range(k)}

        for pair, label in zip(pairs, labels):
            t = int(label)
            # Combine context analysis text and target analysis text for maximum topic coverage
            ctx = pair.get("context_analysis_text") or pair.get("context_text", "")
            tgt = pair.get("target_analysis_text") or pair.get("target_text", "")
            combined_text = f"{ctx} {tgt}".lower()

            words = token_pattern.findall(combined_text)
            filtered = [w for w in words if w not in stop_set and len(w) >= 3]
            topic_words[t].extend(filtered)

        # Document frequency across topics
        word_presence: Dict[str, Set[int]] = {}
        for t in range(k):
            for w in set(topic_words[t]):
                if w not in word_presence:
                    word_presence[w] = set()
                word_presence[w].add(t)

        topic_keywords: Dict[int, List[Dict[str, Any]]] = {}

        for t in range(k):
            total_words = len(topic_words[t])
            if total_words == 0:
                topic_keywords[t] = []
                continue

            tf_counts = Counter(topic_words[t])
            c_tfidf_scores: Dict[str, float] = {}

            for w, cnt in tf_counts.items():
                if cnt < 2 and len(pairs) > 50:
                    continue
                tf = cnt / total_words
                df = len(word_presence.get(w, set()))
                # c-TF-IDF formula
                idf = math.log(1.0 + (k + 1.0) / (df + 1.0)) + 1.0
                c_tfidf_scores[w] = tf * idf

            sorted_vocab = sorted(c_tfidf_scores.items(), key=lambda x: -x[1])[:n_words]
            topic_keywords[t] = [
                {"term": word, "score": round(score, 6), "count": tf_counts[word]}
                for word, score in sorted_vocab
            ]

        return topic_keywords

    def fit(
        self,
        vectors: np.ndarray,
        pairs: Sequence[Dict[str, Any]],
        n_topics: Optional[int] = None,
    ) -> TopicModelResult:
        """
        Fits the topic model on conversational vectors (typically context vectors).
        Extracts content keywords and representative topic exemplars.
        """
        vecs = self._ensure_normalized(vectors)
        if len(vecs) != len(pairs):
            raise ValueError("Length of vectors and pairs must match.")

        eval_history = None
        k = n_topics or self.config.n_topics
        if k is None:
            eval_res = self.evaluate_k(vecs)
            k = eval_res["optimal_k"]
            eval_history = eval_res["k_evaluations"]

        km = KMeans(
            n_clusters=k,
            random_state=self.config.random_seed,
            n_init=self.config.n_init,
            max_iter=self.config.max_iter,
        )
        labels = km.fit_predict(vecs)

        # Normalize centroids
        centroids = km.cluster_centers_.astype(np.float32)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized_centroids = centroids / norms

        sil = float(silhouette_score(vecs, labels, metric="cosine"))
        ch = float(calinski_harabasz_score(vecs, labels))

        # Extract keywords
        topic_keywords = self.extract_topic_keywords(pairs, labels)

        # Extract medoids per topic
        profiles: List[Dict[str, Any]] = []
        total_samples = len(pairs)

        for t in range(k):
            indices = np.where(labels == t)[0]
            size = len(indices)
            pct = round((size / total_samples) * 100, 2) if total_samples > 0 else 0.0

            # Medoid lookup
            t_vecs = vecs[indices]
            sims = np.dot(t_vecs, normalized_centroids[t])
            top_order = np.argsort(-sims)[: self.config.exemplars_per_topic]

            exemplars = [
                {
                    "pair_id": pairs[indices[i]].get("pair_id", ""),
                    "similarity": round(float(sims[i]), 4),
                    "context_text": pairs[indices[i]].get("context_text", "")[:160],
                    "target_text": pairs[indices[i]].get("target_text", "")[:120],
                    "source_file": pairs[indices[i]].get("source_file", ""),
                }
                for i in top_order
            ]

            # Generate intuitive automatic label from top 3 keywords
            kws = topic_keywords.get(t, [])
            label_suffix = "_".join(kw["term"] for kw in kws[:3]) if kws else "general"
            topic_label = f"Topic_{t:02d}_{label_suffix}"

            profiles.append({
                "topic_id": t,
                "topic_label": topic_label,
                "size": size,
                "percentage": pct,
                "keywords": kws,
                "exemplars": exemplars,
            })

        # Sort profiles by size descending
        profiles.sort(key=lambda p: -p["size"])

        return TopicModelResult(
            n_topics=k,
            labels=labels,
            centroids=normalized_centroids,
            silhouette=sil,
            calinski_harabasz=ch,
            topic_profiles=profiles,
            evaluation_history=eval_history,
            metadata={
                "n_samples": len(vecs),
                "dimension": vecs.shape[1],
                "random_seed": self.config.random_seed,
            },
        )

    def compute_style_topic_matrix(
        self,
        topic_labels: np.ndarray,
        style_labels: np.ndarray,
        topic_names: Optional[Dict[int, str]] = None,
        style_names: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """
        Computes the Situational Style Transition Probability Matrix:
        P(Style_j | Topic_i) = count(Topic_i and Style_j) / count(Topic_i)

        Shows how conversational style shifts depending on the domain topic being discussed.
        """
        if len(topic_labels) != len(style_labels):
            raise ValueError("Lengths of topic_labels and style_labels must match.")

        unique_topics = sorted(np.unique(topic_labels))
        unique_styles = sorted(np.unique(style_labels))

        matrix: Dict[str, Dict[str, float]] = {}
        counts_matrix: Dict[str, Dict[str, int]] = {}
        dominant_style_per_topic: Dict[str, Dict[str, Any]] = {}

        for t in unique_topics:
            t_mask = topic_labels == t
            t_count = int(np.sum(t_mask))
            t_name = topic_names.get(int(t), f"Topic_{t:02d}") if topic_names else f"Topic_{t:02d}"

            matrix[t_name] = {}
            counts_matrix[t_name] = {}

            best_style = None
            best_prob = -1.0

            for s in unique_styles:
                s_mask = (style_labels == s) & t_mask
                s_count = int(np.sum(s_mask))
                prob = round(s_count / t_count, 4) if t_count > 0 else 0.0

                s_name = style_names.get(int(s), f"Style_{s:02d}") if style_names else f"Style_{s:02d}"
                matrix[t_name][s_name] = prob
                counts_matrix[t_name][s_name] = s_count

                if prob > best_prob:
                    best_prob = prob
                    best_style = s_name

            dominant_style_per_topic[t_name] = {
                "dominant_style": best_style,
                "probability": best_prob,
                "topic_volume": t_count,
            }

        return {
            "probabilities": matrix,
            "counts": counts_matrix,
            "dominant_styles": dominant_style_per_topic,
        }
