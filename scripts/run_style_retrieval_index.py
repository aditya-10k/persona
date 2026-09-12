"""
Execution script for Stage T028 & T029: Style Embedding Index & Style Retrieval Engine.
Builds the local vector index from the 63 curated style exemplars and
benchmarks selective few-shot retrieval across multiple conversational situations.
Outputs:
- data/processed/style_index/style_index.npz
- data/processed/style_index/style_index_metadata.json
- data/processed/style_index/retrieval_benchmark_report.json
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.persona.retrieval import (
    RetrievalConfig,
    RetrievalResult,
    StyleEmbeddingIndex,
    StyleRetriever,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_style_retrieval_index")


def main() -> None:
    # Ensure stdout handles unicode/emojis on Windows terminals
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

    data_dir = PROJECT_ROOT / "data" / "processed"
    embeddings_dir = data_dir / "embeddings"
    index_dir = data_dir / "style_index"

    examples_path = data_dir / "style_examples.jsonl"
    pairs_path = data_dir / "train_pairs.jsonl"
    target_vecs_path = embeddings_dir / "train_target_vectors.npz"
    ctx_vecs_path = embeddings_dir / "train_context_vectors.npz"

    if not examples_path.exists():
        logger.error("Style examples not found at %s. Run T026-T027 first.", examples_path)
        sys.exit(1)

    # 1. Load style examples
    logger.info("Loading style exemplars from %s ...", examples_path)
    examples = []
    with open(examples_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    logger.info("Loaded %d style exemplars.", len(examples))

    # 2. Load precomputed train embeddings and pairs to map exemplar vectors
    logger.info("Loading training pairs and 768-dim embeddings ...")
    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs = [json.loads(l) for l in f if l.strip()]

    with np.load(target_vecs_path) as npz:
        all_target_vecs = npz["vectors"]
    with np.load(ctx_vecs_path) as npz:
        all_ctx_vecs = npz["vectors"]

    # Match each exemplar to its index in train_pairs
    resp_to_idx = {p["target_text"].strip(): i for i, p in enumerate(pairs)}
    matched_ctx_vecs = []
    matched_resp_vecs = []
    unmatched = 0

    for ex in examples:
        resp_clean = ex["response"].strip()
        if resp_clean in resp_to_idx:
            orig_idx = resp_to_idx[resp_clean]
            matched_ctx_vecs.append(all_ctx_vecs[orig_idx])
            matched_resp_vecs.append(all_target_vecs[orig_idx])
        else:
            unmatched += 1
            # Fallback: zero vector normalized
            v = np.ones(all_ctx_vecs.shape[1], dtype=np.float32)
            matched_ctx_vecs.append(v / np.linalg.norm(v))
            matched_resp_vecs.append(v / np.linalg.norm(v))

    if unmatched > 0:
        logger.warning("%d exemplars could not be mapped to train pairs.", unmatched)
    else:
        logger.info("All %d exemplars successfully mapped to 768-dim vectors.", len(examples))

    ctx_matrix = np.stack(matched_ctx_vecs)
    resp_matrix = np.stack(matched_resp_vecs)

    # 3. Build StyleEmbeddingIndex (T028)
    logger.info("Building StyleEmbeddingIndex ...")
    index = StyleEmbeddingIndex(
        context_vectors=ctx_matrix,
        response_vectors=resp_matrix,
        metadata=examples,
        embedding_model="l3cube-pune/hindi-sentence-bert-nli",
    )

    # 4. Persist index
    logger.info("Saving index to %s ...", index_dir)
    index.save(index_dir)

    # 5. Benchmark StyleRetriever (T029)
    logger.info("Benchmarking StyleRetriever across conversational scenarios ...")
    config = RetrievalConfig(
        top_k=3,
        retrieval_mode="context",
        diversity_lambda=0.70,
        min_similarity_threshold=0.20,
        selective_retrieval=True,
    )
    retriever = StyleRetriever(index, config=config)

    # Test scenarios using vectors from matched categories
    scenarios = [
        {
            "name": "Trivial Greeting (Selective Early Exit)",
            "query_text": "hi",
            "query_vector": None,
            "expected_retrieval": False,
        },
        {
            "name": "One-Word Acknowledgement (Selective Early Exit)",
            "query_text": "ok",
            "query_vector": None,
            "expected_retrieval": False,
        },
        {
            "name": "Pure Emojis (Selective Early Exit)",
            "query_text": "😭😭",
            "query_vector": None,
            "expected_retrieval": False,
        },
        {
            "name": "Technical Collaboration Query",
            "query_text": "bhai curl endpoint run nahi ho raha backend me check kar",
            # Use vector from first technical_collab exemplar
            "query_vector": ctx_matrix[0],
            "expected_retrieval": True,
            "category": "technical_collab",
        },
        {
            "name": "Casual Banter / Hanging Out Query",
            "query_text": "kaha hai bhai chill karra hu movie dekhne chalega",
            # Use vector from first casual_banter exemplar
            "query_vector": ctx_matrix[12],
            "expected_retrieval": True,
            "category": "casual_banter",
        },
        {
            "name": "Conflict & Denial Query",
            "query_text": "tune galat code merge kia tha kya branch me",
            # Use vector from first conflict_friction exemplar
            "query_vector": ctx_matrix[24],
            "expected_retrieval": True,
            "category": "conflict_friction",
        },
        {
            "name": "Career & Placement Query",
            "query_text": "placement shortlisting aai kya company ki",
            # Use vector from first career_academic exemplar
            "query_vector": ctx_matrix[36],
            "expected_retrieval": True,
            "category": "career_academic",
        },
        {
            "name": "Advice & Inquisitive Probing Query",
            "query_text": "kya lagta he bhai konsa framework use karna chaiye",
            # Use vector from first advice_probing exemplar
            "query_vector": ctx_matrix[51],
            "expected_retrieval": True,
            "category": "advice_probing",
        },
    ]

    benchmark_results = []
    logger.info("------------------------------------------------------------")
    logger.info("  SCENARIO RETRIEVAL BENCHMARK")
    logger.info("------------------------------------------------------------")

    for s in scenarios:
        start_t = time.perf_counter()
        res: RetrievalResult = retriever.retrieve(
            query_text=s["query_text"],
            query_vector=s["query_vector"],
            category_filter=s.get("category"),
            top_k=3,
        )
        latency = (time.perf_counter() - start_t) * 1000.0

        benchmark_results.append({
            "scenario": s["name"],
            "query_text": s["query_text"],
            "retrieval_performed": res.retrieval_performed,
            "retrieval_reason": res.retrieval_reason,
            "exemplars_retrieved": len(res.items),
            "top_match_id": res.items[0].example["example_id"] if res.items else None,
            "top_match_sim": res.items[0].similarity_score if res.items else None,
            "latency_ms": round(latency, 3),
        })

        status = "RETRIEVED" if res.retrieval_performed else "EARLY EXIT (SKIPPED)"
        top_sim_str = f"(Top Sim: {res.items[0].similarity_score:.4f})" if res.items else ""
        logger.info(" %-40s | %-20s | %6.3f ms %s",
                    s["name"][:40], status, latency, top_sim_str)

    logger.info("------------------------------------------------------------")
    mean_latency = np.mean([b["latency_ms"] for b in benchmark_results])
    logger.info("Mean Retrieval Latency: %.3f ms", mean_latency)

    # 6. Persist benchmark report
    benchmark_report_path = index_dir / "retrieval_benchmark_report.json"
    with open(benchmark_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "index_size": index.size,
            "index_dimension": index.dimension,
            "embedding_model": index.embedding_model,
            "mean_latency_ms": round(float(mean_latency), 3),
            "benchmark_scenarios": benchmark_results,
            "sample_formatted_prompt": benchmark_results[3],
        }, f, indent=2, ensure_ascii=False)

    logger.info("Saved benchmark report to %s", benchmark_report_path)
    logger.info("Stages T028 & T029 complete. Vector index and selective retriever ready.")


if __name__ == "__main__":
    main()
