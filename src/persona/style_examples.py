"""
Stage T026 & T027: Representative Style Dataset & Algorithmic Selection Engine.
Selects a diverse, non-duplicative, privacy-filtered set of exemplar turns
across situational categories, texting archetypes, and sentence lengths
using multi-axis stratification, centroid typicality, and Maximal Marginal Relevance (MMR).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.nlp.situations import SituationalCategory, SituationalClassifier

logger = logging.getLogger(__name__)

# Cluster archetype labels mapping
CLUSTER_NAMES: Dict[int, str] = {
    0: "Denial_Friction (C00)",
    1: "Venting_Banter (C01)",
    2: "Reactive_Slang (C02)",
    3: "Inquisitive_Probing (C03)",
    4: "Technical_Collab (C04)",
    5: "Minimalist_Confirm (C05)",
}

# Regex patterns for feature tagging
RE_EMOJI = re.compile(
    r"[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf]",
    flags=re.UNICODE,
)
RE_ELONGATION = re.compile(r"([a-zA-Z])\1{2,}")
RE_DEV_SCRIPT = re.compile(r"[\u0900-\u097f]")
RE_LATIN_WORD = re.compile(r"\b[a-zA-Z]+\b")
RE_SLANG = re.compile(
    r"\b(bhai|bhaiii|bro|broo|bc|bkl|chutiya|randi|fati|fck|fuck|damn|legit|fr|ngl|tbh|bruh|lol|lmao|ded|shit|pata|accha|achaa|kuch|nahi|nhi|chodd|chalo|scene|yaar)\b",
    re.IGNORECASE,
)
RE_TERMINAL_PUNCT = re.compile(r"[.!?]+$")
RE_EXCLUDE_TERMS = re.compile(
    r"\b(aakarshit|aarushi|aditya|afroz|ticktickboom|goklu|gooners|heta|karani|rishi|shah|samruddhi|sudhya|triponovaa|akshat|vora|ankit|datta|anupam|tarav|swayam|yash|svkm|gaurav|advaith|manoj)\b",
    re.IGNORECASE,
)


@dataclass
class StyleExampleConfig:
    """Configuration for representative style exemplar selection."""

    target_per_category: int = 12       # 12 examples per situation * 6 situations = 72 total
    min_words: int = 1
    max_words: int = 120
    diversity_lambda: float = 0.65      # MMR weight: 0.65 typicality vs 0.35 diversity
    max_pairwise_similarity: float = 0.88  # Rejects near-duplicate responses
    min_centroid_similarity: float = 0.15  # Rejects outlier responses
    embedding_model: str = "l3cube-pune/hindi-sentence-bert-nli"
    random_seed: int = 42


@dataclass
class StyleExample:
    """Single representative style exemplar."""

    example_id: str
    category: str
    cluster_id: int
    cluster_label: str
    context: str
    response: str
    style_tags: List[str]
    length_bin: str
    word_count: int
    centroid_similarity: float
    source_conversation: str
    pair_id: str
    embedding_model: str = "l3cube-pune/hindi-sentence-bert-nli"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StyleExampleResult:
    """Aggregated output of the representative style dataset."""

    total_examples: int
    examples: List[StyleExample]
    category_counts: Dict[str, int]
    cluster_counts: Dict[str, int]
    length_bin_counts: Dict[str, int]
    mean_centroid_similarity: float
    mean_pairwise_similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_examples": self.total_examples,
            "category_counts": self.category_counts,
            "cluster_counts": self.cluster_counts,
            "length_bin_counts": self.length_bin_counts,
            "mean_centroid_similarity": round(self.mean_centroid_similarity, 4),
            "mean_pairwise_similarity": round(self.mean_pairwise_similarity, 4),
            "metadata": self.metadata,
            "examples": [e.to_dict() for e in self.examples],
        }


def extract_style_tags(text: str, is_multi_bubble: bool = False) -> List[str]:
    """
    Extracts descriptive forensic style tags for a response turn.
    """
    tags: List[str] = []
    clean_text = text.strip()
    words = clean_text.split()

    # 1. Casing
    if clean_text.islower():
        tags.append("all_lowercase")
    elif clean_text.isupper() and len(clean_text) > 2:
        tags.append("all_caps")
    else:
        tags.append("mixed_case")

    # 2. Punctuation
    if not RE_TERMINAL_PUNCT.search(clean_text):
        tags.append("unpunctuated")
    else:
        if "?" in clean_text:
            tags.append("question")
        if "!" in clean_text:
            tags.append("exclamation")
        if any(clean_text.count(p) >= 2 for p in [".", "!", "?"]) or "?!" in clean_text:
            tags.append("multiple_punct")

    # 3. Emojis
    emojis = RE_EMOJI.findall(clean_text)
    if emojis:
        tags.append("emoji_present")
        if len(emojis) >= 2:
            tags.append("emoji_burst")

    # 4. Elongations
    if RE_ELONGATION.search(clean_text):
        tags.append("elongation")

    # 5. Slang
    if RE_SLANG.search(clean_text):
        tags.append("slang_marker")

    # 6. Code-Switching / Script
    has_dev = bool(RE_DEV_SCRIPT.search(clean_text))
    latin_words = [w.lower() for w in RE_LATIN_WORD.findall(clean_text)]
    if has_dev:
        tags.append("devanagari_script")

    # Simple Hinglish vs English dominant
    hindi_markers = {"nahi", "nhi", "kya", "hai", "bhai", "toh", "ka", "ki", "ke", "ko", "se", "kar", "me", "mai", "bhi", "tha", "thi", "the"}
    has_hindi_marker = any(w in hindi_markers for w in latin_words)
    if has_hindi_marker and any(w not in hindi_markers and len(w) > 3 for w in latin_words):
        tags.append("hinglish_mixed")
    elif has_hindi_marker:
        tags.append("hindi_dominant")
    else:
        tags.append("english_dominant")

    # 7. Clause Structure
    if len(words) <= 3 and not any(w in {"is", "are", "hai", "tha", "hoga", "do", "did"} for w in latin_words):
        tags.append("verbless_fragment")
    elif len(words) > 12:
        tags.append("extended_clause")

    # 8. Bubble burstiness
    if is_multi_bubble or "\n" in clean_text:
        tags.append("multi_bubble")
    else:
        tags.append("single_bubble")

    return sorted(list(set(tags)))


def classify_length_bin(word_count: int) -> str:
    """Classifies word count into discrete length bins."""
    if word_count <= 4:
        return "short"
    if word_count <= 14:
        return "medium"
    return "long"


class StyleExampleBuilder:
    """
    Builds a curated, representative style dataset from candidate turns
    and embeddings using multi-axis stratification and MMR selection.
    """

    def __init__(self, config: Optional[StyleExampleConfig] = None):
        self.config = config or StyleExampleConfig()
        self.classifier = SituationalClassifier()

    def _anonymize_id(self, val: str, source_map: Optional[Dict[str, str]] = None) -> str:
        """Anonymizes any raw chat filename prefix in identifiers."""
        if not val:
            return "chat_01:c0001"
        if ":" in val:
            prefix, rest = val.split(":", 1)
            anon_prefix = source_map.get(prefix, "chat_01") if source_map else "chat_01"
            return f"{anon_prefix}:{rest}"
        return source_map.get(val, "chat_01.txt") if source_map else "chat_01.txt"

    def _anonymize_context(self, context_text: str) -> str:
        """Replaces any raw contact names in conversation context headers and @mentions with Friend."""
        if not context_text:
            return ""
        lines = []
        for line in context_text.splitlines():
            if ":" in line:
                sender_part, msg_part = line.split(":", 1)
                sender_part_clean = sender_part.strip()
                clean_msg = re.sub(r"@\w+(?:\s+\w+)?", "@friend", msg_part)
                if sender_part_clean.lower() == "you":
                    lines.append(f"You:{clean_msg}")
                else:
                    lines.append(f"Friend:{clean_msg}")
            else:
                clean_line = re.sub(r"@\w+(?:\s+\w+)?", "@friend", line)
                lines.append(clean_line)
        return "\n".join(lines)

    def _maximal_marginal_relevance(
        self,
        candidate_indices: List[int],
        vectors: np.ndarray,
        centroid: np.ndarray,
        k: int,
        lambda_param: float,
        max_sim: float,
    ) -> List[int]:
        """
        Selects k indices from candidate_indices using Maximal Marginal Relevance.
        Balances centroid similarity (typicality) with inter-example diversity.
        """
        if not candidate_indices or k <= 0:
            return []

        cand_vecs = vectors[candidate_indices]  # Shape: (N, D)
        # Cosine similarity to centroid (vectors and centroid are unit-normalized)
        sim_to_centroid = np.dot(cand_vecs, centroid)

        selected_cand_idx: List[int] = []
        selected_real_idx: List[int] = []

        # Step 1: Select candidate with highest similarity to centroid
        best_first = int(np.argmax(sim_to_centroid))
        selected_cand_idx.append(best_first)
        selected_real_idx.append(candidate_indices[best_first])

        # Step 2: Iteratively select candidates balancing centroid sim and diversity
        while len(selected_real_idx) < min(k, len(candidate_indices)):
            remaining = [i for i in range(len(candidate_indices)) if i not in selected_cand_idx]
            if not remaining:
                break

            rem_vecs = cand_vecs[remaining]
            sel_vecs = cand_vecs[selected_cand_idx]

            # Pairwise similarity between remaining candidates and already selected candidates
            pairwise_sims = np.dot(rem_vecs, sel_vecs.T)  # Shape: (len(rem), len(sel))
            max_sim_to_selected = np.max(pairwise_sims, axis=1)

            # Filter candidates that violate max_pairwise_similarity threshold
            valid_mask = max_sim_to_selected <= max_sim
            if not np.any(valid_mask):
                # Relax slightly if no candidates pass the strict threshold
                valid_mask = np.ones(len(remaining), dtype=bool)

            valid_rem_indices = [idx for idx, valid in enumerate(valid_mask) if valid]
            if not valid_rem_indices:
                break

            # MMR scoring: lambda * sim(x, centroid) - (1 - lambda) * max_y sim(x, y)
            mmr_scores = (
                lambda_param * sim_to_centroid[np.array(remaining)[valid_rem_indices]]
                - (1.0 - lambda_param) * max_sim_to_selected[valid_rem_indices]
            )

            best_match_pos = valid_rem_indices[int(np.argmax(mmr_scores))]
            best_cand_idx = remaining[best_match_pos]

            selected_cand_idx.append(best_cand_idx)
            selected_real_idx.append(candidate_indices[best_cand_idx])

        return selected_real_idx

    def build_exemplars(
        self,
        pairs: Sequence[Dict[str, Any]],
        vectors: np.ndarray,
        cluster_assignments: Optional[Dict[str, int]] = None,
        source_map: Optional[Dict[str, str]] = None,
    ) -> StyleExampleResult:
        """
        Builds the representative style dataset across situations and length bins.

        Args:
            pairs: Sequence of context-response turn dicts.
            vectors: Unit-normalized 2D numpy array of response vectors matching pairs.
            cluster_assignments: Mapping from pair_id to cluster_id (0..5).
            source_map: Optional mapping to anonymize raw chat filenames.
        """
        if len(pairs) != len(vectors):
            raise ValueError(f"Pairs count ({len(pairs)}) must match vectors length ({len(vectors)})")
        if len(pairs) == 0:
            raise ValueError("Input pairs cannot be empty")

        cluster_map = cluster_assignments or {}

        # 1. Classify and annotate each candidate turn
        annotated: List[Dict[str, Any]] = []
        for i, p in enumerate(pairs):
            resp_text = p.get("target_text", "").strip()
            ctx_text = p.get("context_text", "").strip()
            words = resp_text.split()
            word_count = len(words)

            # Filter uninformative or oversized responses, or attachments
            if word_count < self.config.min_words or word_count > self.config.max_words:
                continue
            if not resp_text or not ctx_text:
                continue
            if "vcard" in resp_text.lower() or "vcard" in ctx_text.lower() or "begin:vcard" in resp_text.lower():
                continue

            # Ensure zero personal friend names in target response or context body
            if RE_EXCLUDE_TERMS.search(resp_text):
                continue
            ctx_body_lines = [l.split(":", 1)[1] if ":" in l else l for l in ctx_text.splitlines()]
            if RE_EXCLUDE_TERMS.search(" ".join(ctx_body_lines)):
                continue

            # Classify situation
            sit_res = self.classifier.classify_turn(resp_text, context_text=ctx_text)
            category = sit_res.primary_situation

            pid = p.get("pair_id", f"p_{i}")
            cid = cluster_map.get(pid, 0)
            lbin = classify_length_bin(word_count)

            annotated.append({
                "idx": i,
                "pair": p,
                "pair_id": pid,
                "category": category,
                "cluster_id": cid,
                "length_bin": lbin,
                "word_count": word_count,
                "is_multi_bubble": "\n" in resp_text or p.get("is_initiation", False),
            })

        # 2. Group candidates by situational category and length bin
        by_category_and_bin: Dict[str, Dict[str, List[int]]] = {
            cat.value: {"short": [], "medium": [], "long": []}
            for cat in SituationalCategory
        }

        for item in annotated:
            cat = item["category"]
            lbin = item["length_bin"]
            if cat in by_category_and_bin:
                by_category_and_bin[cat][lbin].append(item["idx"])

        # 3. Stratified selection per situation
        # Allocate quota across length bins: e.g. 12 per category -> 3 short, 5 medium, 4 long
        target_per_cat = self.config.target_per_category
        bin_quotas = {
            "short": max(1, target_per_cat // 4),
            "medium": max(1, target_per_cat // 2),
            "long": max(1, target_per_cat - (target_per_cat // 4) - (target_per_cat // 2)),
        }

        selected_indices: List[int] = []

        for cat, bins in by_category_and_bin.items():
            # Compute category centroid across all candidate turns in this category
            all_cat_indices = bins["short"] + bins["medium"] + bins["long"]
            if not all_cat_indices:
                continue

            cat_vecs = vectors[all_cat_indices]
            cat_centroid = np.mean(cat_vecs, axis=0)
            cat_norm = np.linalg.norm(cat_centroid)
            if cat_norm > 0:
                cat_centroid = cat_centroid / cat_norm

            for lbin, quota in bin_quotas.items():
                cand_pool = bins[lbin]
                if not cand_pool:
                    continue

                # Filter by min_centroid_similarity to reject noise
                pool_vecs = vectors[cand_pool]
                sims = np.dot(pool_vecs, cat_centroid)
                valid_pool = [cand_pool[idx] for idx, s in enumerate(sims) if s >= self.config.min_centroid_similarity]
                if not valid_pool:
                    valid_pool = cand_pool

                chosen = self._maximal_marginal_relevance(
                    candidate_indices=valid_pool,
                    vectors=vectors,
                    centroid=cat_centroid,
                    k=quota,
                    lambda_param=self.config.diversity_lambda,
                    max_sim=self.config.max_pairwise_similarity,
                )
                selected_indices.extend(chosen)

        # 4. Construct StyleExample objects
        # De-duplicate index list while preserving order
        seen_indices: Set[int] = set()
        unique_selected_indices: List[int] = []
        for idx in selected_indices:
            if idx not in seen_indices:
                seen_indices.add(idx)
                unique_selected_indices.append(idx)

        examples: List[StyleExample] = []
        category_counts: Dict[str, int] = Counter()
        cluster_counts: Dict[str, int] = Counter()
        length_bin_counts: Dict[str, int] = Counter()
        centroid_similarities: List[float] = []

        for e_num, idx in enumerate(unique_selected_indices, 1):
            p = pairs[idx]
            resp_text = p.get("target_text", "").strip()
            ctx_text = p.get("context_text", "").strip()
            pid = p.get("pair_id", f"p_{idx}")
            cid = cluster_map.get(pid, 0)
            words = resp_text.split()
            word_count = len(words)
            lbin = classify_length_bin(word_count)

            sit_res = self.classifier.classify_turn(resp_text, context_text=ctx_text)
            category = sit_res.primary_situation

            # Tags
            style_tags = extract_style_tags(resp_text, is_multi_bubble="\n" in resp_text)

            # Anonymize IDs
            anon_pid = self._anonymize_id(pid, source_map)
            conv_id = p.get("conversation_id", "conv_01")
            anon_conv = self._anonymize_id(conv_id, source_map)

            # Centroid similarity of this example to its category centroid
            cat_indices = by_category_and_bin.get(category, {}).get(lbin, [])
            if cat_indices:
                c_vec = np.mean(vectors[cat_indices], axis=0)
                norm = np.linalg.norm(c_vec)
                if norm > 0:
                    c_vec = c_vec / norm
                sim_val = float(np.dot(vectors[idx], c_vec))
            else:
                sim_val = 0.5

            centroid_similarities.append(sim_val)
            category_counts[category] += 1
            cluster_counts[CLUSTER_NAMES.get(cid, f"Cluster_{cid}")] += 1
            length_bin_counts[lbin] += 1

            ex = StyleExample(
                example_id=f"s{e_num:03d}",
                category=category,
                cluster_id=cid,
                cluster_label=CLUSTER_NAMES.get(cid, f"Cluster_{cid}"),
                context=self._anonymize_context(ctx_text),
                response=resp_text,
                style_tags=style_tags,
                length_bin=lbin,
                word_count=word_count,
                centroid_similarity=round(sim_val, 4),
                source_conversation=anon_conv,
                pair_id=anon_pid,
                embedding_model=self.config.embedding_model,
            )
            examples.append(ex)

        # 5. Compute diversity metrics
        if len(unique_selected_indices) > 1:
            sel_vecs = vectors[unique_selected_indices]
            pw_matrix = np.dot(sel_vecs, sel_vecs.T)
            # Upper triangle off-diagonal elements
            triu_indices = np.triu_indices(len(unique_selected_indices), k=1)
            mean_pw_sim = float(np.mean(pw_matrix[triu_indices]))
        else:
            mean_pw_sim = 0.0

        mean_centroid_sim = float(np.mean(centroid_similarities)) if centroid_similarities else 0.0

        return StyleExampleResult(
            total_examples=len(examples),
            examples=examples,
            category_counts=dict(category_counts),
            cluster_counts=dict(cluster_counts),
            length_bin_counts=dict(length_bin_counts),
            mean_centroid_similarity=mean_centroid_sim,
            mean_pairwise_similarity=mean_pw_sim,
            metadata={
                "target_per_category": self.config.target_per_category,
                "diversity_lambda": self.config.diversity_lambda,
                "max_pairwise_similarity": self.config.max_pairwise_similarity,
                "min_centroid_similarity": self.config.min_centroid_similarity,
                "embedding_model": self.config.embedding_model,
            },
        )
