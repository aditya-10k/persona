"""
Stage T006: Text Cleaning / Normalization

Maintains a multi-view representation of each message:
- raw_text: Unmutated sanitized message content (never overwritten)
- normalized_text: Cleaned of invisible unicode / whitespace artifacts, standardized NFC,
  strictly preserving casing, emojis, slang, and punctuation clusters
- tokenized_text: Structured token sequence enriched with style metadata tags
- analysis_text: Semantic representation tailored for Transformer embeddings and topic modeling
"""

import json
import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import emoji

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Produces multi-view text representations and extracts style markers
    while preserving ground truth and sender identity.
    """

    # Invisible unicode control characters
    ZERO_WIDTH_CHARS = re.compile(r'[\u200b\u200c\u200d\ufeff\u200e\u200f]')

    # Whitespace cleanup
    WHITESPACE_RE = re.compile(r'[ \t\f\v]+')
    REDUNDANT_NEWLINES = re.compile(r'\n{3,}')

    # WhatsApp export metadata / media placeholders to strip for analysis_text
    FORWARDED_PREFIX = re.compile(r'(?i)^\[Forwarded\]\s*')
    MEDIA_PLACEHOLDERS = re.compile(r'(?i)<[^>]+omitted>')

    # Elongations (3 or more consecutive identical letters)
    ELONGATION_PATTERN = re.compile(r'([a-zA-Z])\1{2,}')

    # Punctuation clusters (2 or more expressive punctuation marks)
    PUNCT_CLUSTER_PATTERN = re.compile(r'([!?.]{2,})')

    # All-caps words (2 or more uppercase letters, ignoring masking tokens like [URL])
    ALL_CAPS_PATTERN = re.compile(r'^[A-Z]{2,}$')

    # Tokenizer pattern: recognizes [TAGS], words (with apostrophes/hyphens), punct clusters, emojis, or single symbols
    TOKEN_PATTERN = re.compile(
        r'\[[A-Za-z0-9_]+\]|\w+(?:[\'-]\w+)*|[!?.]{2,}|[!?,.:;()"\']|[^\w\s]'
    )

    def normalize_text(self, text: str) -> str:
        """
        NFC Unicode normalization, removes zero-width characters,
        replaces non-breaking spaces, standardizes line breaks, and collapses
        redundant internal horizontal whitespace.
        Strictly preserves casing, emojis, elongations, and punctuation clusters.
        """
        if not text:
            return ""

        # Canonical decomposition followed by canonical composition
        normalized = unicodedata.normalize("NFC", text)

        # Remove invisible formatting characters
        normalized = self.ZERO_WIDTH_CHARS.sub("", normalized)

        # Replace non-breaking space with standard space
        normalized = normalized.replace("\xa0", " ")

        # Standardize line breaks
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse excessive horizontal spaces within each line
        lines = [self.WHITESPACE_RE.sub(" ", line).strip() for line in normalized.split("\n")]
        normalized = "\n".join(lines).strip()

        # Collapse 3+ newlines down to 2
        normalized = self.REDUNDANT_NEWLINES.sub("\n\n", normalized)

        return normalized

    def build_analysis_text(self, normalized_text: str) -> str:
        """
        Creates a clean representation for Transformer embeddings:
        - Strips WhatsApp boilerplate ([Forwarded], <image omitted>, etc.)
        - Detaches emojis with spaces so subword tokenizers treat them as distinct tokens
        - Detaches punctuation clusters
        - Reduces extreme character elongations (runs of 3+ letters reduced to 2)
        - Normalizes whitespace
        - Preserves casing!
        """
        if not normalized_text:
            return ""

        text = normalized_text

        # 1. Strip WhatsApp export boilerplate
        text = self.FORWARDED_PREFIX.sub("", text)
        text = self.MEDIA_PLACEHOLDERS.sub("", text)

        # 2. Detach emojis with spaces
        text = emoji.replace_emoji(text, replace=lambda chars, d: f" {chars} ")

        # 3. Detach punctuation clusters with spaces
        text = self.PUNCT_CLUSTER_PATTERN.sub(r" \1 ", text)

        # 4. Controlled elongation reduction (max 2 identical letters)
        text = self.ELONGATION_PATTERN.sub(r"\1\1", text)

        # 5. Collapse all whitespace into single spaces and strip
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def extract_tokens_and_style(self, normalized_text: str) -> Dict[str, Any]:
        """
        Tokenizes text and extracts style markers for linguistic fingerprinting.
        """
        if not normalized_text:
            return {
                "tokens": [],
                "token_count": 0,
                "character_count": 0,
                "style_markers": {
                    "has_emoji": False,
                    "emojis": [],
                    "emoji_count": 0,
                    "has_elongation": False,
                    "elongated_words": [],
                    "has_all_caps": False,
                    "all_caps_words": [],
                    "has_punct_cluster": False,
                    "punctuation_clusters": [],
                },
            }

        # 1. Extract emojis in order of appearance
        emojis_found = [item["emoji"] for item in emoji.emoji_list(normalized_text)]

        # 2. Separate emojis by spaces for clean token boundary extraction
        spaced_for_tokens = emoji.replace_emoji(
            normalized_text, replace=lambda chars, d: f" {chars} "
        )

        # 3. Tokenize
        tokens = self.TOKEN_PATTERN.findall(spaced_for_tokens)

        # 4. Extract style markers
        elongated_words: List[Dict[str, Any]] = []
        all_caps_words: List[str] = []

        for token in tokens:
            # Check for elongation
            elongation_match = self.ELONGATION_PATTERN.search(token)
            if elongation_match:
                elongated_words.append({
                    "word": token,
                    "base": self.ELONGATION_PATTERN.sub(r"\1\1", token),
                    "char": elongation_match.group(1),
                    "count": len(elongation_match.group(0)),
                })

            # Check for ALL_CAPS words (excluding masking tokens like [URL], [EMAIL])
            if token.startswith("[") and token.endswith("]"):
                continue
            if self.ALL_CAPS_PATTERN.match(token):
                all_caps_words.append(token)

        # 5. Extract expressive punctuation clusters
        punct_clusters = self.PUNCT_CLUSTER_PATTERN.findall(normalized_text)

        return {
            "tokens": tokens,
            "token_count": len(tokens),
            "character_count": len(normalized_text),
            "style_markers": {
                "has_emoji": len(emojis_found) > 0,
                "emojis": emojis_found,
                "emoji_count": len(emojis_found),
                "has_elongation": len(elongated_words) > 0,
                "elongated_words": elongated_words,
                "has_all_caps": len(all_caps_words) > 0,
                "all_caps_words": all_caps_words,
                "has_punct_cluster": len(punct_clusters) > 0,
                "punctuation_clusters": punct_clusters,
            },
        }

    def create_views(self, raw_text: str) -> Dict[str, Any]:
        """
        Creates the 4 distinct views for a given raw message text.
        """
        norm_text = self.normalize_text(raw_text)
        analysis = self.build_analysis_text(norm_text)
        token_info = self.extract_tokens_and_style(norm_text)

        return {
            "raw_text": raw_text,
            "normalized_text": norm_text,
            "analysis_text": analysis,
            "tokenized_text": token_info,
        }

    def clean_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriches a canonical message record with the multi-view dictionary.
        Strictly preserves record["sender"] and all line provenance.
        """
        raw_text = record.get("text", "")
        views = self.create_views(raw_text)

        cleaned = dict(record)
        # Explicitly preserve sender exactly as original
        cleaned["sender"] = record.get("sender")
        # Keep text pointing to normalized_text for backward compatibility
        cleaned["text"] = views["normalized_text"]
        cleaned["views"] = views

        return cleaned


def clean_dataset(
    input_jsonl_path: str,
    output_jsonl_path: str,
    report_output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Streams sanitized_messages.jsonl, computes multi-view representations,
    and writes cleaned_messages.jsonl and cleaning_report.json.
    """
    in_path = Path(input_jsonl_path)
    out_path = Path(output_jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cleaner = TextCleaner()

    total_records = 0
    total_tokens = 0
    messages_with_emojis = 0
    messages_with_elongations = 0
    messages_with_all_caps = 0
    messages_with_punct_clusters = 0

    emoji_counter = Counter()
    elongated_words_counter = Counter()
    all_caps_counter = Counter()

    with open(in_path, "r", encoding="utf-8") as in_f, open(
        out_path, "w", encoding="utf-8"
    ) as out_f:
        for line in in_f:
            line_str = line.strip()
            if not line_str:
                continue

            record = json.loads(line_str)
            total_records += 1

            clean_rec = cleaner.clean_record(record)
            out_f.write(json.dumps(clean_rec, ensure_ascii=False) + "\n")

            style = clean_rec["views"]["tokenized_text"]["style_markers"]
            total_tokens += clean_rec["views"]["tokenized_text"]["token_count"]

            if style["has_emoji"]:
                messages_with_emojis += 1
                for em in style["emojis"]:
                    emoji_counter[em] += 1

            if style["has_elongation"]:
                messages_with_elongations += 1
                for el in style["elongated_words"]:
                    elongated_words_counter[el["word"].lower()] += 1

            if style["has_all_caps"]:
                messages_with_all_caps += 1
                for w in style["all_caps_words"]:
                    all_caps_counter[w] += 1

            if style["has_punct_cluster"]:
                messages_with_punct_clusters += 1

    report = {
        "schema_version": "1.0",
        "task_id": "T006",
        "status": "COMPLETE",
        "total_messages_processed": total_records,
        "total_tokens_extracted": total_tokens,
        "avg_tokens_per_message": round(total_tokens / total_records, 2) if total_records else 0,
        "style_distributions": {
            "messages_with_emojis": messages_with_emojis,
            "messages_with_elongations": messages_with_elongations,
            "messages_with_all_caps": messages_with_all_caps,
            "messages_with_punct_clusters": messages_with_punct_clusters,
            "emoji_message_percentage": round(messages_with_emojis / total_records * 100, 2) if total_records else 0,
            "elongation_message_percentage": round(messages_with_elongations / total_records * 100, 2) if total_records else 0,
            "all_caps_message_percentage": round(messages_with_all_caps / total_records * 100, 2) if total_records else 0,
        },
        "top_15_emojis": dict(emoji_counter.most_common(15)),
        "top_15_elongated_words": dict(elongated_words_counter.most_common(15)),
        "top_15_all_caps_words": dict(all_caps_counter.most_common(15)),
    }

    if report_output_path:
        rep_path = Path(report_output_path)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rep_path, "w", encoding="utf-8") as rep_f:
            json.dump(report, rep_f, indent=2, ensure_ascii=False)

    return report
