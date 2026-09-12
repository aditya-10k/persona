"""
Stage T013: Semantic Similarity Analysis Runner
Executes comprehensive semantic similarity analysis on train_pairs dataset:
- Response Diversity Index
- Context-Response Alignment (Mirroring vs. Reacting)
- Real-time Top-K Retrieval on sample prompts
Generates data/processed/similarity_report.json.
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

# Ensure UTF-8 on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.nlp.embeddings import EmbeddingConfig, EmbeddingEngine
from src.nlp.similarity import SimilarityEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("similarity_analysis")


def load_pairs(filepath: Path) -> List[Dict[str, Any]]:
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def run_similarity_analysis():
    logger.info("Initializing Stage T013 Semantic Similarity Analysis...")

    embeddings_dir = PROJECT_ROOT / "data" / "processed" / "embeddings"
    target_npz = embeddings_dir / "train_target_vectors.npz"
    context_npz = embeddings_dir / "train_context_vectors.npz"
    pairs_file = PROJECT_ROOT / "data" / "processed" / "train_pairs.jsonl"
    report_file = PROJECT_ROOT / "data" / "processed" / "similarity_report.json"

    if not target_npz.exists() or not context_npz.exists():
        logger.error("Embedding vectors not found in %s. Please run T012 first.", embeddings_dir)
        sys.exit(1)

    # 1. Load Pre-computed Embeddings & Dataset
    logger.info("Loading pre-computed embeddings and training pairs...")
    target_data = np.load(target_npz)
    target_vectors = target_data["vectors"]

    context_data = np.load(context_npz)
    context_vectors = context_data["vectors"]

    pairs = load_pairs(pairs_file)
    pair_ids = [p["pair_id"] for p in pairs]
    total_pairs = len(pairs)

    logger.info("Loaded %d vectors of dimension %d.", total_pairs, target_vectors.shape[1])

    # Initialize Embedding Engine for live query encoding
    emb_cfg = EmbeddingConfig(model_name="l3cube-pune/hindi-sentence-bert-nli", batch_size=32)
    embedding_engine = EmbeddingEngine(emb_cfg)

    # Initialize Similarity Engines
    target_sim_engine = SimilarityEngine(
        vectors=target_vectors,
        pair_ids=pair_ids,
        pairs_data=pairs,
        embedding_engine=embedding_engine,
    )
    context_sim_engine = SimilarityEngine(
        vectors=context_vectors,
        pair_ids=pair_ids,
        pairs_data=pairs,
        embedding_engine=embedding_engine,
    )

    # 2. Compute Response Diversity Index
    logger.info("Computing Response Diversity Index across training responses...")
    t0_div = time.time()
    diversity_stats = target_sim_engine.compute_diversity_index(sample_size=1000, random_seed=42)
    t_div_duration = time.time() - t0_div
    logger.info(
        "Diversity Index: %.4f (Mean Pairwise Sim: %.4f) computed in %.2fs",
        diversity_stats["diversity_index"],
        diversity_stats["mean_pairwise_similarity"],
        t_div_duration,
    )

    # 3. Compute Context-to-Response Alignment (Mirroring vs. Reacting)
    logger.info("Computing Context-to-Response Alignment distribution...")
    alignment_stats = SimilarityEngine.compute_context_response_alignment(
        context_vectors=context_vectors,
        target_vectors=target_vectors,
    )
    logger.info(
        "Mean Alignment: %.4f | Mirroring: %.1f%% | Balanced: %.1f%% | Reactive: %.1f%%",
        alignment_stats["mean_alignment"],
        alignment_stats["behavior_breakdown"]["mirroring_ratio"] * 100,
        alignment_stats["behavior_breakdown"]["balanced_ratio"] * 100,
        alignment_stats["behavior_breakdown"]["reactive_ratio"] * 100,
    )

    # 4. Run Sample Retrieval Benchmark across 4 Intent Types
    test_queries = [
        {
            "intent": "Tech / Coding / Debugging",
            "query": "python me code optimize kaise kare",
        },
        {
            "intent": "Casual Banter / Meeting Plans",
            "query": "bhai kal milte hai sham ko kya bolta hai",
        },
        {
            "intent": "Emotional Venting / Empathy",
            "query": "bohot stress ho raha hai dimag kaam nahi kar raha",
        },
        {
            "intent": "Skepticism / Opinion",
            "query": "are you sure ye kaam karega?",
        },
    ]

    retrieval_benchmarks = []
    logger.info("\n--- EXECUTING TOP-K RETRIEVAL BENCHMARKS ---")
    for item in test_queries:
        q = item["query"]
        t0_q = time.time()
        # Search against historical contexts (find when someone said something similar to query)
        matches = context_sim_engine.find_similar(query=q, top_k=3)
        q_latency_ms = (time.time() - t0_q) * 1000

        top_matches_data = [m.to_dict() for m in matches]
        retrieval_benchmarks.append({
            "intent": item["intent"],
            "query": q,
            "latency_ms": round(q_latency_ms, 2),
            "top_matches": top_matches_data,
        })

        print(f"\n[Query: '{q}' ({item['intent']}) - Latency: {q_latency_ms:.2f}ms]")
        for i, m in enumerate(matches, 1):
            print(f"  #{i} [Score: {m.score:.4f}] Context: '{m.context_text[:60]}' -> Your Response: '{m.target_text[:60]}'")

    # 5. Export Report
    report = {
        "stage": "T013",
        "task": "Semantic Similarity",
        "embedding_model": emb_cfg.model_name,
        "embedding_dimension": target_vectors.shape[1],
        "total_pairs_analyzed": total_pairs,
        "response_diversity": diversity_stats,
        "context_response_alignment": alignment_stats,
        "sample_retrieval_benchmarks": retrieval_benchmarks,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("\nReport successfully saved to %s", report_file)
    print("\n--- STAGE T013 SUMMARY ---")
    print(f"Total Pairs: {total_pairs}")
    print(f"Response Diversity Index: {diversity_stats['diversity_index']} (High diversity > 0.70)")
    print(f"Context-Response Alignment: Mean = {alignment_stats['mean_alignment']}")
    print(f"  - Mirroring (Direct Answers): {alignment_stats['behavior_breakdown']['mirroring_ratio']*100:.1f}%")
    print(f"  - Balanced Conversational:  {alignment_stats['behavior_breakdown']['balanced_ratio']*100:.1f}%")
    print(f"  - Reactive (Pivots/Emojis):   {alignment_stats['behavior_breakdown']['reactive_ratio']*100:.1f}%")


if __name__ == "__main__":
    run_similarity_analysis()
