"""
Execution script for Stage T026 & T027: Representative Style Dataset & Selection Engine.
Extracts a diverse, non-duplicative, privacy-filtered exemplar dataset
from training pairs across 6 situations and 3 length bins using MMR selection.
Outputs:
- data/processed/style_examples.jsonl
- data/processed/style_examples_report.json
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

from src.persona.style_examples import (
    StyleExampleBuilder,
    StyleExampleConfig,
    StyleExampleResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_style_examples")


def main() -> None:
    # Ensure stdout handles unicode/emojis on Windows terminals
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

    data_dir = PROJECT_ROOT / "data" / "processed"
    embeddings_dir = data_dir / "embeddings"
    clustering_dir = data_dir / "clustering"

    pairs_path = data_dir / "train_pairs.jsonl"
    target_vecs_path = embeddings_dir / "train_target_vectors.npz"
    assignments_path = clustering_dir / "cluster_assignments.jsonl"

    output_jsonl = data_dir / "style_examples.jsonl"
    output_report = data_dir / "style_examples_report.json"

    if not pairs_path.exists() or not target_vecs_path.exists():
        logger.error("Required training pairs or target vectors not found.")
        sys.exit(1)

    # 1. Load training pairs
    logger.info("Loading training pairs from %s ...", pairs_path)
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    # 2. Load target response vectors
    logger.info("Loading target response vectors from %s ...", target_vecs_path)
    with np.load(target_vecs_path) as npz:
        target_vectors = npz["vectors"]

    # 3. Load cluster assignments
    cluster_map = {}
    if assignments_path.exists():
        logger.info("Loading cluster assignments from %s ...", assignments_path)
        with open(assignments_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cluster_map[rec["pair_id"]] = rec["cluster_id"]
    else:
        logger.warning("Cluster assignments not found at %s; defaulting to 0.", assignments_path)

    # 4. Build deterministic anonymous source map
    unique_sources = sorted(list({p.get("source_file", "") for p in pairs if p.get("source_file")}))
    source_map = {}
    for i, s in enumerate(unique_sources, 1):
        source_map[s] = f"chat_{i:02d}.txt"
        # Also map conversation ID prefix
        source_map[s.replace(".txt", "")] = f"chat_{i:02d}"

    logger.info("Anonymization source map built for %d unique sources.", len(source_map))

    # 5. Build representative exemplars
    config = StyleExampleConfig(
        target_per_category=12,         # 12 per situation * 6 = 72 target
        diversity_lambda=0.65,          # 65% typicality, 35% diversity
        max_pairwise_similarity=0.88,   # Reject near-duplicates
        min_centroid_similarity=0.15,   # Filter outliers
        random_seed=42,
    )
    builder = StyleExampleBuilder(config)

    logger.info("Selecting representative style exemplars via MMR ...")
    start_time = time.perf_counter()
    result: StyleExampleResult = builder.build_exemplars(
        pairs=pairs,
        vectors=target_vectors,
        cluster_assignments=cluster_map,
        source_map=source_map,
    )
    elapsed = time.perf_counter() - start_time
    logger.info("Selection completed in %.2f seconds.", elapsed)

    # 6. Log selection breakdown
    logger.info("============================================================")
    logger.info("  REPRESENTATIVE STYLE EXAMPLAR SELECTION SUMMARY")
    logger.info("============================================================")
    logger.info("Total Exemplars Selected: %d", result.total_examples)
    logger.info("Mean Typicality (Centroid Sim): %.4f", result.mean_centroid_similarity)
    logger.info("Mean Pairwise Redundancy (Sim): %.4f", result.mean_pairwise_similarity)
    logger.info("------------------------------------------------------------")
    logger.info("By Situation Category:")
    for cat, cnt in sorted(result.category_counts.items(), key=lambda x: -x[1]):
        logger.info("  - %-22s: %2d exemplars", cat, cnt)
    logger.info("------------------------------------------------------------")
    logger.info("By Texting Archetype (Cluster):")
    for clus, cnt in sorted(result.cluster_counts.items(), key=lambda x: -x[1]):
        logger.info("  - %-26s: %2d exemplars", clus, cnt)
    logger.info("------------------------------------------------------------")
    logger.info("By Length Bin:")
    for lbin, cnt in sorted(result.length_bin_counts.items(), key=lambda x: -x[1]):
        logger.info("  - %-10s: %2d exemplars", lbin, cnt)
    logger.info("============================================================")

    # 7. Persist artifacts
    logger.info("Writing exemplars to %s ...", output_jsonl)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for ex in result.examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")

    logger.info("Writing report to %s ...", output_report)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info("Stage T026 & T027 complete. %d style exemplars ready.", result.total_examples)


if __name__ == "__main__":
    main()
