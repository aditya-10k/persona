"""
Stage T010: Sampling Strategy

Implements multi-dimensional stratified sampling across:
- Source chat files / interlocutor relationships (caps large sources, retains 100% of small sources)
- Response length bins (short: 1-3 words, medium: 4-15 words, long: 16+ words)
- Interaction modes (preserves 100% of conversation initiation pairs)
- Temporal timeline (even sampling across chronological history)
Ensures deterministic reproducibility via a fixed random seed.
"""

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class StratifiedSampler:
    """
    Performs multi-dimensional stratified sampling over context-response pairs.
    """

    def __init__(self, max_per_source: int = 500, random_seed: int = 42):
        self.max_per_source = max_per_source
        self.random_seed = random_seed

    @staticmethod
    def classify_length_bin(text: str) -> Tuple[str, int]:
        """
        Categorizes response length into short, medium, or long based on word count.
        """
        words = text.strip().split()
        count = len(words)
        if count <= 3:
            return "short", count
        elif count <= 15:
            return "medium", count
        else:
            return "long", count

    def sample_source_pairs(
        self, source_file: str, pairs: List[Dict[str, Any]], rng: random.Random
    ) -> List[Dict[str, Any]]:
        """
        Stratified sampling within a single source file.
        - Retains 100% of initiation pairs
        - If total <= max_per_source, retains 100%
        - Otherwise, stratifies reply pairs evenly across length bins
        """
        if len(pairs) <= self.max_per_source:
            # Undersized source: preserve 100%
            sampled = []
            for p in pairs:
                l_bin, w_cnt = self.classify_length_bin(p.get("target_text", ""))
                rec = dict(p)
                rec["sampling_strata"] = {
                    "source_file": source_file,
                    "length_bin": l_bin,
                    "word_count": w_cnt,
                    "is_initiation": p.get("is_initiation", False),
                }
                sampled.append(rec)
            return sampled

        # Separate initiations vs replies
        initiations: List[Dict[str, Any]] = []
        replies: List[Dict[str, Any]] = []

        for p in pairs:
            if p.get("is_initiation", False):
                initiations.append(p)
            else:
                replies.append(p)

        # Always keep 100% of initiations
        sampled_pairs: List[Dict[str, Any]] = []
        for p in initiations:
            l_bin, w_cnt = self.classify_length_bin(p.get("target_text", ""))
            rec = dict(p)
            rec["sampling_strata"] = {
                "source_file": source_file,
                "length_bin": l_bin,
                "word_count": w_cnt,
                "is_initiation": True,
            }
            sampled_pairs.append(rec)

        # Calculate remaining quota for replies
        reply_quota = max(0, self.max_per_source - len(sampled_pairs))
        if reply_quota <= 0:
            return sampled_pairs

        # Group replies by length bin
        length_bins: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for p in replies:
            l_bin, _ = self.classify_length_bin(p.get("target_text", ""))
            length_bins[l_bin].append(p)

        # Allocate quota evenly across non-empty length bins
        active_bins = [b for b in ["short", "medium", "long"] if length_bins[b]]
        if not active_bins:
            return sampled_pairs

        # Proportional or equal allocation
        quota_per_bin = reply_quota // len(active_bins)
        remainder = reply_quota % len(active_bins)

        sampled_replies: List[Dict[str, Any]] = []
        for idx, b in enumerate(active_bins):
            bin_items = length_bins[b]
            bin_quota = quota_per_bin + (1 if idx < remainder else 0)

            if len(bin_items) <= bin_quota:
                chosen = bin_items
            else:
                # Sample systematically across chronological timeline for even temporal spread
                indices = sorted(rng.sample(range(len(bin_items)), bin_quota))
                chosen = [bin_items[i] for i in indices]

            for p in chosen:
                l_bin, w_cnt = self.classify_length_bin(p.get("target_text", ""))
                rec = dict(p)
                rec["sampling_strata"] = {
                    "source_file": source_file,
                    "length_bin": l_bin,
                    "word_count": w_cnt,
                    "is_initiation": False,
                }
                sampled_replies.append(rec)

        sampled_pairs.extend(sampled_replies)
        return sampled_pairs

    def sample_dataset(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Executes multi-dimensional stratified sampling across all source files.
        """
        rng = random.Random(self.random_seed)

        # Group by source file
        by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for p in pairs:
            by_source[p.get("source_file", "unknown")].append(p)

        all_sampled: List[Dict[str, Any]] = []
        for src_file in sorted(by_source.keys()):
            src_pairs = by_source[src_file]
            sampled_src = self.sample_source_pairs(src_file, src_pairs, rng)
            all_sampled.extend(sampled_src)

        return all_sampled


def sample_context_pairs_dataset(
    input_pairs_path: str,
    output_pairs_path: str,
    report_output_path: Optional[str] = None,
    max_per_source: int = 500,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Streams filtered_pairs.jsonl, samples a balanced corpus,
    and writes sampled_pairs.jsonl and sampling_report.json.
    """
    in_path = Path(input_pairs_path)
    out_path = Path(output_pairs_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sampler = StratifiedSampler(max_per_source=max_per_source, random_seed=random_seed)

    all_pairs: List[Dict[str, Any]] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                all_pairs.append(json.loads(line_str))

    sampled_pairs = sampler.sample_dataset(all_pairs)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for p in sampled_pairs:
            out_f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Aggregate distribution metrics (before vs after)
    src_before = Counter(p.get("source_file", "unknown") for p in all_pairs)
    src_after = Counter(p.get("source_file", "unknown") for p in sampled_pairs)

    len_before = Counter(
        StratifiedSampler.classify_length_bin(p.get("target_text", ""))[0] for p in all_pairs
    )
    len_after = Counter(p["sampling_strata"]["length_bin"] for p in sampled_pairs)

    init_before = sum(1 for p in all_pairs if p.get("is_initiation"))
    init_after = sum(1 for p in sampled_pairs if p["sampling_strata"]["is_initiation"])

    report = {
        "schema_version": "1.0",
        "task_id": "T010",
        "status": "COMPLETE",
        "parameters": {
            "max_per_source": max_per_source,
            "random_seed": random_seed,
        },
        "total_pairs_before": len(all_pairs),
        "total_pairs_after": len(sampled_pairs),
        "sampling_ratio_percentage": round(len(sampled_pairs) / len(all_pairs) * 100, 2) if all_pairs else 0,
        "source_distribution": {
            src: {
                "before": src_before[src],
                "before_pct": round(src_before[src] / len(all_pairs) * 100, 2),
                "after": src_after[src],
                "after_pct": round(src_after[src] / len(sampled_pairs) * 100, 2),
            }
            for src in sorted(src_before.keys())
        },
        "length_bin_distribution": {
            l_bin: {
                "before": len_before[l_bin],
                "before_pct": round(len_before[l_bin] / len(all_pairs) * 100, 2),
                "after": len_after[l_bin],
                "after_pct": round(len_after[l_bin] / len(sampled_pairs) * 100, 2),
            }
            for l_bin in ["short", "medium", "long"]
        },
        "initiations": {
            "before": init_before,
            "after": init_after,
            "retention_percentage": round(init_after / init_before * 100, 2) if init_before else 0,
        },
    }

    if report_output_path:
        rep_path = Path(report_output_path)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rep_path, "w", encoding="utf-8") as rep_f:
            json.dump(report, rep_f, indent=2, ensure_ascii=False)

    return report
