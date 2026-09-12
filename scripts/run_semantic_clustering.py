"""
Execution script for Stage T014: Semantic Clustering & Archetype Discovery.
Loads train target vectors (2,750 x 768) and pairs, evaluates optimal K,
performs Spherical K-Means clustering, extracts representative exemplars,
profiles archetypes, and saves artifacts to data/processed/clustering/.
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

from src.nlp.clustering import ClusterConfig, ClusteringEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_semantic_clustering")


def main() -> None:
    # Ensure stdout handles emojis on Windows terminals
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

    data_dir = PROJECT_ROOT / "data" / "processed"
    embeddings_dir = data_dir / "embeddings"
    clustering_dir = data_dir / "clustering"
    clustering_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = embeddings_dir / "train_target_vectors.npz"
    metadata_path = embeddings_dir / "train_target_metadata.json"
    pairs_path = data_dir / "train_pairs.jsonl"

    if not vectors_path.exists() or not pairs_path.exists():
        logger.error("Required training vectors or pairs file not found.")
        sys.exit(1)

    # 1. Load data
    logger.info("Loading train target vectors from %s ...", vectors_path)
    with np.load(vectors_path) as npz:
        vectors = npz["vectors"]

    logger.info("Loading metadata from %s ...", metadata_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        pair_ids = meta.get("pair_ids", [])

    logger.info("Loading training pairs from %s ...", pairs_path)
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    logger.info("Loaded %d vectors (dim=%d) and %d pairs.", len(vectors), vectors.shape[1], len(pairs))
    assert len(vectors) == len(pair_ids) == len(pairs), "Data length mismatch!"

    # 2. Evaluate K across range [6, 18]
    k_range = list(range(6, 19))
    logger.info("Evaluating candidate cluster counts K in range %s ...", k_range)

    engine = ClusteringEngine(ClusterConfig(random_seed=42, exemplars_per_cluster=5))

    start_eval = time.perf_counter()
    eval_results = engine.evaluate_k(vectors, k_range=k_range)
    eval_duration = time.perf_counter() - start_eval

    logger.info("K evaluation finished in %.2f seconds.", eval_duration)
    logger.info("------------------------------------------------------------")
    logger.info("  K  | Silhouette (Cosine) | Calinski-Harabasz | Inertia    ")
    logger.info("------------------------------------------------------------")
    for k in k_range:
        m = eval_results["k_evaluations"][k]
        marker = " <-- [BEST SILHOUETTE]" if k == eval_results["optimal_k"] else ""
        logger.info(" %2d  |      %.4f         |     %8.2f      |  %8.2f%s",
                    k, m["silhouette"], m["calinski_harabasz"], m["inertia"], marker)
    logger.info("------------------------------------------------------------")

    optimal_k = eval_results["optimal_k"]
    logger.info("Optimal cluster count selected: K = %d (Silhouette = %.4f)",
                optimal_k, eval_results["best_silhouette"])

    # 3. Fit optimal model
    logger.info("Fitting Spherical K-Means with K = %d ...", optimal_k)
    start_fit = time.perf_counter()
    result = engine.fit(vectors, k=optimal_k)
    fit_duration = time.perf_counter() - start_fit
    logger.info("Model fitted in %.2f seconds. Final Silhouette = %.4f, CH = %.2f",
                fit_duration, result.silhouette, result.calinski_harabasz)

    # 4. Extract exemplars (medoids)
    logger.info("Extracting top 5 real medoid exemplars per cluster ...")
    exemplars = engine.extract_exemplars(
        vectors, result.labels, pair_ids, top_n=5, centroids=result.centroids
    )

    # 5. Profile clusters
    logger.info("Profiling cluster archetypes (c-TF-IDF, word counts, emojis) ...")
    profiles = engine.profile_clusters(
        pairs, result.labels, exemplars=exemplars, top_vocab_n=10, top_emojis_n=5
    )

    # 6. Save cluster assignments
    assignments_path = clustering_dir / "cluster_assignments.jsonl"
    logger.info("Writing cluster assignments to %s ...", assignments_path)
    pairs_by_id = {p["pair_id"]: p for p in pairs}

    with open(assignments_path, "w", encoding="utf-8") as f:
        for idx, (pid, label) in enumerate(zip(pair_ids, result.labels)):
            c = int(label)
            # Compute cosine similarity of this vector to its cluster centroid
            sim = float(np.dot(vectors[idx], result.centroids[c]))
            p = pairs_by_id[pid]
            record = {
                "pair_id": pid,
                "cluster_id": c,
                "similarity_to_centroid": round(sim, 4),
                "target_text": p.get("target_text", ""),
                "source_file": p.get("source_file", ""),
                "is_initiation": p.get("is_initiation", False),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 7. Save cluster profiles
    profiles_path = clustering_dir / "cluster_profiles.json"
    logger.info("Writing cluster profiles to %s ...", profiles_path)
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

    # 8. Save comprehensive clustering report
    report_path = clustering_dir / "clustering_report.json"
    logger.info("Writing clustering report to %s ...", report_path)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "total_pairs": len(pairs),
            "vector_dimension": vectors.shape[1],
            "split": "train",
        },
        "k_sweep_evaluation": eval_results["k_evaluations"],
        "optimal_k": optimal_k,
        "silhouette_score": round(result.silhouette, 4),
        "calinski_harabasz_index": round(result.calinski_harabasz, 2),
        "inertia": round(result.inertia, 2) if result.inertia is not None else None,
        "cluster_size_distribution": {
            str(p["cluster_id"]): {
                "size": p["size"],
                "percentage": p["percentage"],
                "mean_words": p["target_word_stats"]["mean"],
                "top_terms": [v["term"] for v in p["distinctive_vocabulary"][:5]],
                "top_emojis": [e["emoji"] for e in p["emoji_signature"][:3]],
            }
            for p in profiles
        },
        "timings": {
            "k_evaluation_seconds": round(eval_duration, 2),
            "fit_seconds": round(fit_duration, 2),
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 9. Print human-readable summary of discovered archetypes
    print("\n" + "=" * 80)
    print(f"STAGE T014: SEMANTIC CLUSTERING REPORT (K = {optimal_k})")
    print(f"Silhouette Score: {result.silhouette:.4f} | Calinski-Harabasz: {result.calinski_harabasz:.2f}")
    print("=" * 80)
    for p in profiles:
        cid = p["cluster_id"]
        sz = p["size"]
        pct = p["percentage"]
        w_mean = p["target_word_stats"]["mean"]
        top_words = ", ".join(f"{v['term']} ({v['count']})" for v in p["distinctive_vocabulary"][:5])
        top_emojis = " ".join(e["emoji"] for e in p["emoji_signature"][:3]) or "None"

        print(f"\n--- Cluster {cid:02d} | Size: {sz} ({pct}%) | Avg Words: {w_mean:.1f} | Emojis: {top_emojis} ---")
        print(f"    Key Vocab: {top_words}")
        if p["exemplars"]:
            medoid = p["exemplars"][0]
            cleaned_target = medoid["target_text"].replace("\n", " ")[:90]
            print(f"    Medoid ({medoid['similarity']:.4f}): \"{cleaned_target}\"")
    print("\n" + "=" * 80)
    logger.info("Stage T014 Semantic Clustering complete!")


if __name__ == "__main__":
    main()
