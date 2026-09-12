"""
Stage T016: Surface Linguistic Profiler (Linguistic Fingerprinting)
Measures empirical quantitative distributions of word counts, burstiness,
lexical diversity (TTR, Guiraud, Hapax Legomena), casing conventions,
zero-punctuation cadence, emoji burst patterns, and Hinglish orthographic quirks.
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

logger = logging.getLogger(__name__)

# Common Hinglish conversational abbreviations to monitor
COMMON_ABBREVIATIONS: Set[str] = {
    "clg", "grp", "oa", "sgpa", "cgpa", "bkl", "bc", "whi", "woto", "islie",
    "pr", "ts", "js", "ml", "ai", "ds", "sem", "yr", "plz", "pls", "btw",
    "idk", "tbh", "ngl", "fr", "atb", "thx", "thnx", "tym", "bday", "msg",
    "pic", "vid", "re", "ig", "abt", "diff", "info", "admin", "repo", "acc",
}

# Regex patterns for surface linguistic markers
RE_ELONGATION = re.compile(r"\b[a-zA-Z]*([a-zA-Z])\1{2,}[a-zA-Z]*\b", re.IGNORECASE)
RE_TERMINAL_PUNCT = re.compile(r"[.?!]+$")
RE_QUESTION = re.compile(r"\?")
RE_MULTI_QUESTION = re.compile(r"\?{2,}")
RE_EXCLAMATION = re.compile(r"!")
RE_MULTI_EXCLAMATION = re.compile(r"!{2,}")
RE_ELLIPSIS = re.compile(r"\.{2,}")
RE_WORDS = re.compile(r"[a-zA-Z\u0900-\u097F0-9_']+")


def _compute_distribution_stats(values: Sequence[float | int]) -> Dict[str, float]:
    """Computes standard summary statistics for a numerical sequence."""
    if not values:
        return {
            "mean": 0.0, "median": 0.0, "std": 0.0,
            "min": 0.0, "max": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0,
        }
    arr = np.array(values, dtype=float)
    return {
        "mean": round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "std": round(float(np.std(arr)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "p25": round(float(np.percentile(arr, 25)), 2),
        "p75": round(float(np.percentile(arr, 75)), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
    }


@dataclass
class LinguisticProfile:
    """Quantitative surface linguistic profile for a corpus or archetype."""

    total_turns: int
    total_messages: int
    total_words: int
    total_characters: int

    # Volumetrics & Lengths
    words_per_turn: Dict[str, float]
    chars_per_turn: Dict[str, float]
    burstiness: Dict[str, Any]

    # Lexical Diversity
    lexical_diversity: Dict[str, float]

    # Casing & Capitalization
    casing_distribution: Dict[str, float]

    # Punctuation & Terminal Mechanics
    punctuation_metrics: Dict[str, float]

    # Emoji Signatures
    emoji_fingerprint: Dict[str, Any]

    # Hinglish Morphological Quirks & Slang
    orthographic_quirks: Dict[str, Any]

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_turns": self.total_turns,
            "total_messages": self.total_messages,
            "total_words": self.total_words,
            "total_characters": self.total_characters,
            "words_per_turn": self.words_per_turn,
            "chars_per_turn": self.chars_per_turn,
            "burstiness": self.burstiness,
            "lexical_diversity": self.lexical_diversity,
            "casing_distribution": self.casing_distribution,
            "punctuation_metrics": self.punctuation_metrics,
            "emoji_fingerprint": self.emoji_fingerprint,
            "orthographic_quirks": self.orthographic_quirks,
            "metadata": self.metadata,
        }


class LinguisticProfiler:
    """
    Extracts surface linguistic features, lexical diversity indices,
    punctuation mechanics, casing styles, and Hinglish orthographic quirks.
    """

    def analyze_turn(self, turn_text: str, message_count: int = 1) -> Dict[str, Any]:
        """
        Extracts raw surface linguistic features from a single conversational turn.
        """
        raw_text = turn_text.strip()
        words = RE_WORDS.findall(raw_text.lower())
        char_count = len(raw_text)
        word_count = len(words)

        # Casing analysis
        alpha_chars = [c for c in raw_text if c.isalpha()]
        is_all_lower = len(alpha_chars) > 0 and all(c.islower() for c in alpha_chars)
        is_all_upper = len(alpha_chars) > 1 and all(c.isupper() for c in alpha_chars)
        has_initial_cap = len(alpha_chars) > 0 and alpha_chars[0].isupper()

        # Punctuation analysis
        has_terminal_punct = bool(RE_TERMINAL_PUNCT.search(raw_text))
        has_question = bool(RE_QUESTION.search(raw_text))
        has_multi_question = bool(RE_MULTI_QUESTION.search(raw_text))
        has_exclamation = bool(RE_EXCLAMATION.search(raw_text))
        has_multi_exclamation = bool(RE_MULTI_EXCLAMATION.search(raw_text))
        has_ellipsis = bool(RE_ELLIPSIS.search(raw_text))

        # Emoji analysis
        em_list = emoji.emoji_list(raw_text)
        emoji_count = len(em_list)
        emojis = [e["emoji"] for e in em_list]
        is_emoji_only = emoji_count > 0 and len(raw_text.replace(" ", "")) == sum(len(e) for e in emojis)
        has_emoji_burst = any(cnt >= 2 for cnt in Counter(emojis).values())

        # Orthographic quirks
        has_elongation = bool(RE_ELONGATION.search(raw_text))
        elongated_matches = [m.group(0).lower() for m in RE_ELONGATION.finditer(raw_text)]

        abbreviations = [w for w in words if w in COMMON_ABBREVIATIONS]

        return {
            "char_count": char_count,
            "word_count": word_count,
            "message_count": message_count,
            "words": words,
            "is_all_lower": is_all_lower,
            "is_all_upper": is_all_upper,
            "has_initial_cap": has_initial_cap,
            "has_terminal_punct": has_terminal_punct,
            "has_question": has_question,
            "has_multi_question": has_multi_question,
            "has_exclamation": has_exclamation,
            "has_multi_exclamation": has_multi_exclamation,
            "has_ellipsis": has_ellipsis,
            "emoji_count": emoji_count,
            "emojis": emojis,
            "is_emoji_only": is_emoji_only,
            "has_emoji_burst": has_emoji_burst,
            "has_elongation": has_elongation,
            "elongations": elongated_matches,
            "abbreviations": abbreviations,
        }

    def fit(self, pairs: Sequence[Dict[str, Any]], name: str = "global") -> LinguisticProfile:
        """
        Aggregates turn-level linguistic features into a comprehensive LinguisticProfile.
        """
        n_turns = len(pairs)
        if n_turns == 0:
            raise ValueError("Pairs list cannot be empty for linguistic profiling.")

        word_counts: List[int] = []
        char_counts: List[int] = []
        message_counts: List[int] = []
        all_tokens: List[str] = []

        all_lower_count = 0
        all_upper_count = 0
        initial_cap_count = 0

        terminal_punct_count = 0
        question_count = 0
        multi_question_count = 0
        exclamation_count = 0
        multi_exclamation_count = 0
        ellipsis_count = 0

        total_emojis = 0
        turns_with_emojis = 0
        emoji_only_turns = 0
        emoji_burst_turns = 0
        emoji_counter: Counter[str] = Counter()

        elongation_turns = 0
        elongation_counter: Counter[str] = Counter()
        abbreviation_counter: Counter[str] = Counter()

        for p in pairs:
            raw_text = p.get("target_text", "")
            # Get message count from target_response or default to 1
            msg_cnt = p.get("target_response", {}).get("message_count", 1) or 1
            # Fallback if text contains explicit newlines indicating message breaks
            if msg_cnt == 1 and "\n" in raw_text:
                # Keep recorded message_count from segmentation if available
                pass

            feats = self.analyze_turn(raw_text, message_count=msg_cnt)

            word_counts.append(feats["word_count"])
            char_counts.append(feats["char_count"])
            message_counts.append(feats["message_count"])
            all_tokens.extend(feats["words"])

            if feats["is_all_lower"]:
                all_lower_count += 1
            if feats["is_all_upper"]:
                all_upper_count += 1
            if feats["has_initial_cap"]:
                initial_cap_count += 1

            if feats["has_terminal_punct"]:
                terminal_punct_count += 1
            if feats["has_question"]:
                question_count += 1
            if feats["has_multi_question"]:
                multi_question_count += 1
            if feats["has_exclamation"]:
                exclamation_count += 1
            if feats["has_multi_exclamation"]:
                multi_exclamation_count += 1
            if feats["has_ellipsis"]:
                ellipsis_count += 1

            if feats["emoji_count"] > 0:
                turns_with_emojis += 1
                total_emojis += feats["emoji_count"]
                emoji_counter.update(feats["emojis"])
            if feats["is_emoji_only"]:
                emoji_only_turns += 1
            if feats["has_emoji_burst"]:
                emoji_burst_turns += 1

            if feats["has_elongation"]:
                elongation_turns += 1
                elongation_counter.update(feats["elongations"])

            abbreviation_counter.update(feats["abbreviations"])

        total_words = len(all_tokens)
        total_chars = sum(char_counts)
        total_msgs = sum(message_counts)

        # Lexical Diversity calculations
        vocab_counter = Counter(all_tokens)
        vocab_size = len(vocab_counter)
        hapax_legomena = sum(1 for cnt in vocab_counter.values() if cnt == 1)
        dis_legomena = sum(1 for cnt in vocab_counter.values() if cnt == 2)

        ttr = round(vocab_size / total_words, 4) if total_words > 0 else 0.0
        root_ttr = round(vocab_size / math.sqrt(total_words), 2) if total_words > 0 else 0.0
        corrected_ttr = round(vocab_size / math.sqrt(2 * total_words), 2) if total_words > 0 else 0.0
        bilog_ttr = round(math.log(vocab_size) / math.log(total_words), 4) if total_words > 1 and vocab_size > 1 else 0.0
        hapax_ratio = round(hapax_legomena / vocab_size, 4) if vocab_size > 0 else 0.0

        # Burstiness (messages per turn)
        msg_arr = np.array(message_counts)
        single_msg_turns = int(np.sum(msg_arr == 1))
        double_msg_turns = int(np.sum(msg_arr == 2))
        multi_msg_turns = int(np.sum(msg_arr >= 3))

        burstiness = {
            "mean_messages_per_turn": round(float(np.mean(msg_arr)), 2),
            "median_messages_per_turn": round(float(np.median(msg_arr)), 2),
            "max_burst_messages": int(np.max(msg_arr)),
            "single_bubble_ratio": round(single_msg_turns / n_turns, 4),
            "double_bubble_ratio": round(double_msg_turns / n_turns, 4),
            "multi_bubble_ratio": round(multi_msg_turns / n_turns, 4),
        }

        # Casing ratios
        casing = {
            "all_lowercase_ratio": round(all_lower_count / n_turns, 4),
            "initial_capital_ratio": round(initial_cap_count / n_turns, 4),
            "all_uppercase_ratio": round(all_upper_count / n_turns, 4),
            "mixed_casing_ratio": round((n_turns - all_lower_count - all_upper_count) / n_turns, 4),
        }

        # Punctuation ratios
        punctuation = {
            "zero_terminal_punctuation_ratio": round((n_turns - terminal_punct_count) / n_turns, 4),
            "question_turns_ratio": round(question_count / n_turns, 4),
            "multi_question_ratio": round(multi_question_count / n_turns, 4),
            "exclamation_turns_ratio": round(exclamation_count / n_turns, 4),
            "multi_exclamation_ratio": round(multi_exclamation_count / n_turns, 4),
            "ellipsis_turns_ratio": round(ellipsis_count / n_turns, 4),
            "punctuation_density_per_100_words": round(
                (terminal_punct_count + question_count + exclamation_count + ellipsis_count) / max(total_words, 1) * 100, 2
            ),
        }

        # Emoji metrics
        emoji_metrics = {
            "turns_with_emojis_ratio": round(turns_with_emojis / n_turns, 4),
            "emoji_only_turns_ratio": round(emoji_only_turns / n_turns, 4),
            "emoji_burst_ratio": round(emoji_burst_turns / max(turns_with_emojis, 1), 4),
            "emojis_per_100_words": round(total_emojis / max(total_words, 1) * 100, 2),
            "top_emojis": [
                {"emoji": em, "count": cnt, "per_turn_ratio": round(cnt / n_turns, 4)}
                for em, cnt in emoji_counter.most_common(15)
            ],
        }

        # Orthographic quirks
        orthography = {
            "elongation_turns_ratio": round(elongation_turns / n_turns, 4),
            "top_elongations": [
                {"token": tok, "count": cnt} for tok, cnt in elongation_counter.most_common(10)
            ],
            "top_abbreviations": [
                {"abbreviation": abb, "count": cnt, "per_turn_ratio": round(cnt / n_turns, 4)}
                for abb, cnt in abbreviation_counter.most_common(12)
            ],
        }

        return LinguisticProfile(
            total_turns=n_turns,
            total_messages=total_msgs,
            total_words=total_words,
            total_characters=total_chars,
            words_per_turn=_compute_distribution_stats(word_counts),
            chars_per_turn=_compute_distribution_stats(char_counts),
            burstiness=burstiness,
            lexical_diversity={
                "vocabulary_size": vocab_size,
                "ttr": ttr,
                "root_ttr_guiraud": root_ttr,
                "corrected_ttr_carroll": corrected_ttr,
                "bilogarithmic_ttr": bilog_ttr,
                "hapax_legomena_count": hapax_legomena,
                "hapax_ratio": hapax_ratio,
                "dis_legomena_count": dis_legomena,
            },
            casing_distribution=casing,
            punctuation_metrics=punctuation,
            emoji_fingerprint=emoji_metrics,
            orthographic_quirks=orthography,
            metadata={"profile_name": name},
        )

    def analyze_by_style(
        self,
        pairs: Sequence[Dict[str, Any]],
        style_labels: np.ndarray,
        style_names: Optional[Dict[int, str]] = None,
    ) -> Dict[str, LinguisticProfile]:
        """
        Computes separate linguistic profiles for each communication archetype (C00 to C05).
        """
        if len(pairs) != len(style_labels):
            raise ValueError("Lengths of pairs and style_labels must match.")

        unique_styles = sorted(np.unique(style_labels))
        profiles: Dict[str, LinguisticProfile] = {}

        for s in unique_styles:
            s_mask = style_labels == s
            s_pairs = [pairs[i] for i in range(len(pairs)) if s_mask[i]]
            s_name = style_names.get(int(s), f"Style_{s:02d}") if style_names else f"Style_{s:02d}"
            profiles[s_name] = self.fit(s_pairs, name=s_name)

        return profiles
