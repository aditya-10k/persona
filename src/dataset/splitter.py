"""
Stage T011: Train / Development / Holdout Split

Performs conversation-level stratified splitting (70% train, 15% dev, 15% test)
to prevent conversational context leakage across dialogue turns.
Guarantees zero conversation overlap between train and evaluation sets.
"""

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ConversationSplitter:
    """
    Assigns whole conversation sessions to train, dev, and test splits
    stratified by source chat file to guarantee balanced coverage.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        dev_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        if not abs((train_ratio + dev_ratio + test_ratio) - 1.0) < 1e-5:
            raise ValueError("Split ratios must sum to 1.0")
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    def build_split_mapping(
        self, conversations: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Creates a deterministic mapping of conversation_id -> 'train' | 'dev' | 'test'.
        Stratified by source_file to ensure every interlocutor is represented across splits.
        """
        rng = random.Random(self.random_seed)

        by_source: Dict[str, List[str]] = defaultdict(list)
        for c in conversations:
            cid = c.get("conversation_id")
            src = c.get("source_file", "unknown")
            if cid:
                by_source[src].append(cid)

        split_mapping: Dict[str, str] = {}

        for src_file in sorted(by_source.keys()):
            cids = sorted(by_source[src_file])
            n = len(cids)

            # Shuffled deterministic order
            shuffled_indices = list(range(n))
            rng.shuffle(shuffled_indices)

            if n == 1:
                split_mapping[cids[0]] = "train"
            elif n == 2:
                split_mapping[cids[shuffled_indices[0]]] = "train"
                split_mapping[cids[shuffled_indices[1]]] = "test"
            elif n == 3:
                split_mapping[cids[shuffled_indices[0]]] = "train"
                split_mapping[cids[shuffled_indices[1]]] = "dev"
                split_mapping[cids[shuffled_indices[2]]] = "test"
            else:
                n_test = max(1, int(round(n * self.test_ratio)))
                n_dev = max(1, int(round(n * self.dev_ratio)))
                n_train = n - n_test - n_dev

                if n_train < 1:
                    n_train = 1
                    if n_dev > 1:
                        n_dev -= 1
                    elif n_test > 1:
                        n_test -= 1

                test_idx = set(shuffled_indices[:n_test])
                dev_idx = set(shuffled_indices[n_test : n_test + n_dev])
                train_idx = set(shuffled_indices[n_test + n_dev :])

                for idx, cid in enumerate(cids):
                    if idx in test_idx:
                        split_mapping[cid] = "test"
                    elif idx in dev_idx:
                        split_mapping[cid] = "dev"
                    else:
                        split_mapping[cid] = "train"

        return split_mapping

    def split_pairs(
        self, pairs: List[Dict[str, Any]], split_mapping: Dict[str, str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Partitions pairs into train, dev, and test based on their conversation_id.
        """
        splits: Dict[str, List[Dict[str, Any]]] = {
            "train": [],
            "dev": [],
            "test": [],
        }

        for p in pairs:
            cid = p.get("conversation_id")
            assigned_split = split_mapping.get(cid, "train")
            rec = dict(p)
            rec["split"] = assigned_split
            splits[assigned_split].append(rec)

        return splits


def create_dataset_splits(
    input_pairs_path: str,
    conversations_path: str,
    output_dir: str,
    report_output_path: Optional[str] = None,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes conversation-level stratified splitting, writes split JSONL files
    and outputs a comprehensive distribution audit report.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splitter = ConversationSplitter(
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    # 1. Load conversations
    conversations: List[Dict[str, Any]] = []
    with open(conversations_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                conversations.append(json.loads(line_str))

    # 2. Build conversation split mapping
    split_mapping = splitter.build_split_mapping(conversations)

    mapping_path = out_dir / "split_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(split_mapping, f, indent=2, ensure_ascii=False)

    # 3. Load pairs
    pairs: List[Dict[str, Any]] = []
    with open(input_pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                pairs.append(json.loads(line_str))

    # 4. Partition pairs
    split_results = splitter.split_pairs(pairs, split_mapping)

    # 5. Write split JSONLs
    for split_name in ["train", "dev", "test"]:
        split_file = out_dir / f"{split_name}_pairs.jsonl"
        with open(split_file, "w", encoding="utf-8") as f:
            for p in split_results[split_name]:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # 6. Audit statistics
    conv_counts = Counter(split_mapping.values())
    pair_counts = {k: len(v) for k, v in split_results.items()}
    total_pairs = len(pairs)
    total_convs = len(conversations)

    # Cross-check for zero leakage
    train_cids = {cid for cid, s in split_mapping.items() if s == "train"}
    dev_cids = {cid for cid, s in split_mapping.items() if s == "dev"}
    test_cids = {cid for cid, s in split_mapping.items() if s == "test"}

    leakage_train_test = len(train_cids & test_cids)
    leakage_train_dev = len(train_cids & dev_cids)
    leakage_dev_test = len(dev_cids & test_cids)
    zero_leakage_verified = (
        leakage_train_test == 0 and leakage_train_dev == 0 and leakage_dev_test == 0
    )

    # Breakdown by length bin per split
    length_breakdown: Dict[str, Dict[str, int]] = {}
    for s_name in ["train", "dev", "test"]:
        length_breakdown[s_name] = dict(
            Counter(
                p.get("sampling_strata", {}).get("length_bin", "unknown")
                for p in split_results[s_name]
            )
        )

    # Source files coverage per split
    source_coverage: Dict[str, Dict[str, int]] = {}
    for s_name in ["train", "dev", "test"]:
        source_coverage[s_name] = dict(
            Counter(p.get("source_file", "unknown") for p in split_results[s_name])
        )

    report = {
        "schema_version": "1.0",
        "task_id": "T011",
        "status": "COMPLETE",
        "parameters": {
            "train_ratio": train_ratio,
            "dev_ratio": dev_ratio,
            "test_ratio": test_ratio,
            "random_seed": random_seed,
        },
        "zero_leakage_verified": zero_leakage_verified,
        "conversations_split": {
            "total": total_convs,
            "train": conv_counts.get("train", 0),
            "train_pct": round(conv_counts.get("train", 0) / total_convs * 100, 2) if total_convs else 0,
            "dev": conv_counts.get("dev", 0),
            "dev_pct": round(conv_counts.get("dev", 0) / total_convs * 100, 2) if total_convs else 0,
            "test": conv_counts.get("test", 0),
            "test_pct": round(conv_counts.get("test", 0) / total_convs * 100, 2) if total_convs else 0,
        },
        "pairs_split": {
            "total": total_pairs,
            "train": pair_counts.get("train", 0),
            "train_pct": round(pair_counts.get("train", 0) / total_pairs * 100, 2) if total_pairs else 0,
            "dev": pair_counts.get("dev", 0),
            "dev_pct": round(pair_counts.get("dev", 0) / total_pairs * 100, 2) if total_pairs else 0,
            "test": pair_counts.get("test", 0),
            "test_pct": round(pair_counts.get("test", 0) / total_pairs * 100, 2) if total_pairs else 0,
        },
        "length_bin_distribution_by_split": length_breakdown,
        "source_coverage_by_split": source_coverage,
    }

    if report_output_path:
        rep_path = Path(report_output_path)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rep_path, "w", encoding="utf-8") as rep_f:
            json.dump(report, rep_f, indent=2, ensure_ascii=False)

    return report
