"""
Execution script for Stage T012: Transformer Embedding Layer.
Generates and persists contextual dense semantic representations
for train_pairs.jsonl into compressed .npz and metadata JSON.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp.embeddings import EmbeddingConfig, EmbeddingEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_embeddings")


def load_pairs(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL pairs dataset."""
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def run_pipeline():
    # Force UTF-8 on Windows
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    input_file = PROJECT_ROOT / "data" / "processed" / "train_pairs.jsonl"
    output_dir = PROJECT_ROOT / "data" / "processed" / "embeddings"
    report_file = output_dir / "embeddings_report.json"

    if not input_file.exists():
        logger.error("Input file %s not found. Please run T011 splitter first.", input_file)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training pairs from %s...", input_file)
    pairs = load_pairs(input_file)
    total_pairs = len(pairs)
    logger.info("Loaded %d training pairs. Initializing embedding engine...", total_pairs)

    config = EmbeddingConfig(
        model_name="l3cube-pune/hindi-sentence-bert-nli",
        batch_size=64,
        device="cpu",
        normalize_embeddings=True,
    )
    engine = EmbeddingEngine(config)

    # 1. Embed Target Responses (What "You" said)
    logger.info("Encoding target responses using view 'target_analysis_text'...")
    t0_targets = time.time()
    target_vectors, target_pair_ids = engine.encode_pairs(
        pairs=pairs,
        text_field="target_analysis_text",
        fallback_field="target_text",
        show_progress_bar=True,
    )
    t_targets_duration = time.time() - t0_targets
    logger.info(
        "Target responses encoded in %.2fs (%.1f pairs/s). Shape: %s",
        t_targets_duration,
        total_pairs / max(t_targets_duration, 0.001),
        target_vectors.shape,
    )

    # Save target embeddings
    target_npz, target_meta = engine.save_embeddings(
        output_dir=output_dir,
        name_prefix="train_target",
        vectors=target_vectors,
        pair_ids=target_pair_ids,
        extra_metadata={
            "split": "train",
            "field_embedded": "target_analysis_text",
            "generation_duration_seconds": round(t_targets_duration, 3),
        },
    )

    # 2. Embed Context Prompts (What the conversation partner said)
    logger.info("Encoding conversation contexts using view 'context_analysis_text'...")
    t0_contexts = time.time()
    context_vectors, context_pair_ids = engine.encode_pairs(
        pairs=pairs,
        text_field="context_analysis_text",
        fallback_field="context_text",
        show_progress_bar=True,
    )
    t_contexts_duration = time.time() - t0_contexts
    logger.info(
        "Contexts encoded in %.2fs (%.1f pairs/s). Shape: %s",
        t_contexts_duration,
        total_pairs / max(t_contexts_duration, 0.001),
        context_vectors.shape,
    )

    # Save context embeddings
    context_npz, context_meta = engine.save_embeddings(
        output_dir=output_dir,
        name_prefix="train_context",
        vectors=context_vectors,
        pair_ids=context_pair_ids,
        extra_metadata={
            "split": "train",
            "field_embedded": "context_analysis_text",
            "generation_duration_seconds": round(t_contexts_duration, 3),
        },
    )

    # 3. Quality & Statistics Verification
    target_norms = np.linalg.norm(target_vectors, axis=1)
    context_norms = np.linalg.norm(context_vectors, axis=1)

    target_nan_count = int(np.isnan(target_vectors).sum())
    context_nan_count = int(np.isnan(context_vectors).sum())

    # Size on disk
    target_size_kb = round(target_npz.stat().st_size / 1024, 2)
    context_size_kb = round(context_npz.stat().st_size / 1024, 2)

    report = {
        "stage": "T012",
        "task": "Transformer Embedding Layer",
        "embedding_model": config.model_name,
        "embedding_dimension": engine.dimension,
        "device": config.device,
        "total_pairs_encoded": total_pairs,
        "target_embeddings": {
            "file": str(target_npz.name),
            "shape": list(target_vectors.shape),
            "size_kb": target_size_kb,
            "nan_count": target_nan_count,
            "mean_l2_norm": float(np.mean(target_norms)),
            "min_l2_norm": float(np.min(target_norms)),
            "max_l2_norm": float(np.max(target_norms)),
            "duration_seconds": round(t_targets_duration, 3),
        },
        "context_embeddings": {
            "file": str(context_npz.name),
            "shape": list(context_vectors.shape),
            "size_kb": context_size_kb,
            "nan_count": context_nan_count,
            "mean_l2_norm": float(np.mean(context_norms)),
            "min_l2_norm": float(np.min(context_norms)),
            "max_l2_norm": float(np.max(context_norms)),
            "duration_seconds": round(t_contexts_duration, 3),
        },
        "total_pipeline_duration_seconds": round(t_targets_duration + t_contexts_duration, 3),
        "holdout_test_quarantined": True,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Embedding generation completed successfully.")
    logger.info("Target vectors: %s (%.1f KB)", target_npz.name, target_size_kb)
    logger.info("Context vectors: %s (%.1f KB)", context_npz.name, context_size_kb)
    logger.info("Report written to %s", report_file.name)
    print("\n--- STAGE T012 SUMMARY ---")
    print(f"Model: {config.model_name}")
    print(f"Dimension: {engine.dimension}")
    print(f"Encoded Pairs: {total_pairs}")
    print(f"Target Vectors Size: {target_size_kb} KB")
    print(f"Context Vectors Size: {context_size_kb} KB")
    print(f"Target NaN count: {target_nan_count}")
    print(f"Context NaN count: {context_nan_count}")
    print(f"Total Time: {t_targets_duration + t_contexts_duration:.2f}s")


if __name__ == "__main__":
    run_pipeline()
