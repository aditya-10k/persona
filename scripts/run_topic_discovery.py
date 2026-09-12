"""
Execution script for Stage T015: Topic Discovery & Situational Style Tagging.
Loads train context vectors, pairs, and T014 style assignments; evaluates optimal
topic count; discovers domain topics via content-specific c-TF-IDF; computes the
P(Style | Topic) transition matrix; and persists artifacts to data/processed/topics/.
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

from src.nlp.topics import TopicConfig, TopicDiscoveryEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_topic_discovery")

# Mapping of T014 cluster IDs to human-readable archetype names
STYLE_NAMES = {
    0: "Denial_Friction (C00)",
    1: "Venting_Banter (C01)",
    2: "Reactive_Slang (C02)",
    3: "Inquisitive_Probing (C03)",
    4: "Technical_Collab (C04)",
    5: "Minimalist_Confirm (C05)",
}


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
    topics_dir = data_dir / "topics"
    topics_dir.mkdir(parents=True, exist_ok=True)

    context_vecs_path = embeddings_dir / "train_context_vectors.npz"
    pairs_path = data_dir / "train_pairs.jsonl"
    assignments_path = clustering_dir / "cluster_assignments.jsonl"

    if not context_vecs_path.exists() or not pairs_path.exists() or not assignments_path.exists():
        logger.error("Required context vectors, pairs, or cluster assignments not found.")
        sys.exit(1)

    # 1. Load context vectors
    logger.info("Loading train context vectors from %s ...", context_vecs_path)
    with np.load(context_vecs_path) as npz:
        context_vectors = npz["vectors"]

    # 2. Load training pairs
    logger.info("Loading training pairs from %s ...", pairs_path)
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    # 3. Load T014 style assignments
    logger.info("Loading style assignments from %s ...", assignments_path)
    style_labels_by_pid = {}
    with open(assignments_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                style_labels_by_pid[rec["pair_id"]] = rec["cluster_id"]

    style_labels = np.array([style_labels_by_pid[p["pair_id"]] for p in pairs], dtype=int)
    logger.info("Loaded %d pairs, %d vectors, and %d style labels.",
                len(pairs), len(context_vectors), len(style_labels))

    # 4. Sweep topic candidate counts
    engine = TopicDiscoveryEngine(TopicConfig(random_seed=42, exemplars_per_topic=4))
    k_range = list(range(5, 12))
    logger.info("Evaluating topic candidate count K in range %s ...", k_range)

    start_eval = time.perf_counter()
    eval_res = engine.evaluate_k(context_vectors, k_range=k_range)
    eval_duration = time.perf_counter() - start_eval

    logger.info("Topic K-evaluation completed in %.2f seconds.", eval_duration)
    logger.info("------------------------------------------------------------")
    logger.info("  K  | Silhouette (Cosine) | Calinski-Harabasz | Inertia    ")
    logger.info("------------------------------------------------------------")
    for k in k_range:
        m = eval_res["k_evaluations"][k]
        marker = " <-- [OPTIMAL]" if k == eval_res["optimal_k"] else ""
        logger.info(" %2d  |      %.4f         |     %8.2f      |  %8.2f%s",
                    k, m["silhouette"], m["calinski_harabasz"], m["inertia"], marker)
    logger.info("------------------------------------------------------------")

    optimal_k = eval_res["optimal_k"]
    logger.info("Selected optimal topic count: K = %d (Silhouette = %.4f)",
                optimal_k, eval_res["best_silhouette"])

    # 5. Fit optimal topic model
    logger.info("Fitting topic discovery model with K = %d ...", optimal_k)
    start_fit = time.perf_counter()
    result = engine.fit(context_vectors, pairs, n_topics=optimal_k)
    fit_duration = time.perf_counter() - start_fit
    logger.info("Topic model fitted in %.2f seconds. Final Silhouette = %.4f",
                fit_duration, result.silhouette)

    # 6. Compute P(Style | Topic) transition matrix
    topic_name_map = {p["topic_id"]: p["topic_label"] for p in result.topic_profiles}
    logger.info("Computing P(Style | Topic) situational transition matrix ...")
    matrix_res = engine.compute_style_topic_matrix(
        result.labels,
        style_labels,
        topic_names=topic_name_map,
        style_names=STYLE_NAMES,
    )

    # 7. Write topic assignments
    topic_assignments_path = topics_dir / "topic_assignments.jsonl"
    logger.info("Writing topic assignments to %s ...", topic_assignments_path)
    with open(topic_assignments_path, "w", encoding="utf-8") as f:
        for idx, (p, t_label, s_label) in enumerate(zip(pairs, result.labels, style_labels)):
            t = int(t_label)
            # Similarity of context vector to topic centroid
            sim = float(np.dot(context_vectors[idx], result.centroids[t]))
            record = {
                "pair_id": p.get("pair_id", ""),
                "topic_id": t,
                "topic_label": topic_name_map.get(t, f"Topic_{t:02d}"),
                "similarity_to_topic_centroid": round(sim, 4),
                "style_id": int(s_label),
                "style_name": STYLE_NAMES.get(int(s_label), f"Style_{s_label:02d}"),
                "source_file": p.get("source_file", ""),
                "target_text": p.get("target_text", ""),
                "context_text": p.get("context_text", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 8. Write topic profiles
    profiles_path = topics_dir / "topic_profiles.json"
    logger.info("Writing topic profiles to %s ...", profiles_path)
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(result.topic_profiles, f, indent=2, ensure_ascii=False)

    # 9. Write topic-style transition matrix
    matrix_path = topics_dir / "topic_style_matrix.json"
    logger.info("Writing topic-style transition matrix to %s ...", matrix_path)
    with open(matrix_path, "w", encoding="utf-8") as f:
        json.dump(matrix_res, f, indent=2, ensure_ascii=False)

    # 10. Write comprehensive topic report
    report_path = topics_dir / "topic_report.json"
    logger.info("Writing topic report to %s ...", report_path)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "total_pairs": len(pairs),
            "vector_dimension": context_vectors.shape[1],
            "split": "train",
        },
        "k_sweep_evaluation": eval_res["k_evaluations"],
        "optimal_k": optimal_k,
        "silhouette_score": round(result.silhouette, 4),
        "calinski_harabasz_index": round(result.calinski_harabasz, 2),
        "topics_summary": [
            {
                "topic_id": p["topic_id"],
                "topic_label": p["topic_label"],
                "size": p["size"],
                "percentage": p["percentage"],
                "top_keywords": [kw["term"] for kw in p["keywords"][:6]],
                "dominant_style": matrix_res["dominant_styles"][p["topic_label"]]["dominant_style"],
                "dominant_style_prob": matrix_res["dominant_styles"][p["topic_label"]]["probability"],
            }
            for p in result.topic_profiles
        ],
        "timings": {
            "evaluation_seconds": round(eval_duration, 2),
            "fit_seconds": round(fit_duration, 2),
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 11. Print human-readable summary
    print("\n" + "=" * 80)
    print(f"STAGE T015: TOPIC DISCOVERY REPORT (K = {optimal_k})")
    print(f"Silhouette Score: {result.silhouette:.4f} | Calinski-Harabasz: {result.calinski_harabasz:.2f}")
    print("=" * 80)

    for p in result.topic_profiles:
        t_id = p["topic_id"]
        t_lbl = p["topic_label"]
        sz = p["size"]
        pct = p["percentage"]
        kws = ", ".join(kw["term"] for kw in p["keywords"][:5])
        dom_style = matrix_res["dominant_styles"][t_lbl]["dominant_style"]
        dom_prob = matrix_res["dominant_styles"][t_lbl]["probability"]

        print(f"\n--- [{t_lbl}] | Size: {sz} ({pct}%) ---")
        print(f"    Domain Keywords: {kws}")
        print(f"    Dominant Style: {dom_style} ({dom_prob * 100:.1f}%)")
        if p["exemplars"]:
            ex = p["exemplars"][0]
            clean_ctx = ex["context_text"].replace("\n", " ")[:80]
            clean_tgt = ex["target_text"].replace("\n", " ")[:80]
            print(f"    Context Trigger: \"{clean_ctx}\"")
            print(f"    Your Response:   \"{clean_tgt}\"")

    print("\n" + "=" * 80)
    logger.info("Stage T015 Topic Discovery complete!")


if __name__ == "__main__":
    main()
