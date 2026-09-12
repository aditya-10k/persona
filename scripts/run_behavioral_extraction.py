"""
Execution script for Stage T021: Behavioral Pattern Extraction.
Quantifies 10 psychological and behavioral traits across 2,750 training turns:
directness, verbosity, hedging, disagreement style, humor, question initiative,
confidence assertion, emotional expressiveness, code-switching propensity, and technical depth.
Mines top supporting exemplars for each trait and saves artifacts to data/processed/behavior/.
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

from src.nlp.behavior import BehavioralExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_behavioral_extraction")

STYLE_NAMES = {
    0: "Denial_Friction (C00)",
    1: "Venting_Banter (C01)",
    2: "Reactive_Slang (C02)",
    3: "Inquisitive_Probing (C03)",
    4: "Technical_Collab (C04)",
    5: "Minimalist_Confirm (C05)",
}


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

    data_dir = PROJECT_ROOT / "data" / "processed"
    clustering_dir = data_dir / "clustering"
    beh_dir = data_dir / "behavior"
    beh_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = data_dir / "train_pairs.jsonl"
    assignments_path = clustering_dir / "cluster_assignments.jsonl"

    if not pairs_path.exists() or not assignments_path.exists():
        logger.error("Required pairs or cluster assignments file not found.")
        sys.exit(1)

    # 1. Load pairs and style labels
    logger.info("Loading training pairs from %s ...", pairs_path)
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    logger.info("Loading cluster assignments from %s ...", assignments_path)
    style_labels_by_pid = {}
    with open(assignments_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                style_labels_by_pid[rec["pair_id"]] = rec["cluster_id"]

    style_labels = np.array([style_labels_by_pid[p["pair_id"]] for p in pairs], dtype=int)
    logger.info("Loaded %d training turns and %d style assignments.", len(pairs), len(style_labels))

    # 2. Extract Global Behavioral Profile
    extractor = BehavioralExtractor()
    logger.info("Computing Global Behavioral Profile across %d turns ...", len(pairs))
    start_time = time.perf_counter()
    global_profile = extractor.extract_profile(pairs, name="global_corpus", max_exemplars=5)
    duration = time.perf_counter() - start_time
    logger.info("Global behavioral extraction completed in %.2f seconds.", duration)

    # 3. Extract Per-Archetype Behavioral Profiles
    logger.info("Computing per-archetype behavioral breakdowns for the 6 styles ...")
    style_profiles = {}
    style_groups = {}
    for p, l in zip(pairs, style_labels):
        style_groups.setdefault(int(l), []).append(p)

    for label, group_pairs in sorted(style_groups.items()):
        name = STYLE_NAMES.get(label, f"Style_{label}")
        style_profiles[name] = extractor.extract_profile(group_pairs, name=name, max_exemplars=3)

    # 4. Save Global Profile
    global_profile_path = beh_dir / "global_behavioral_profile.json"
    logger.info("Writing global behavioral profile to %s ...", global_profile_path)
    with open(global_profile_path, "w", encoding="utf-8") as f:
        json.dump(global_profile.to_dict(), f, indent=2, ensure_ascii=False)

    # 5. Save Per-Style Profiles
    style_profiles_path = beh_dir / "style_behavioral_profiles.json"
    logger.info("Writing style behavioral profiles to %s ...", style_profiles_path)
    style_profiles_dict = {name: prof.to_dict() for name, prof in style_profiles.items()}
    with open(style_profiles_path, "w", encoding="utf-8") as f:
        json.dump(style_profiles_dict, f, indent=2, ensure_ascii=False)

    # 6. Save Comprehensive Behavioral Report
    report_path = beh_dir / "behavioral_report.json"
    logger.info("Writing behavioral summary report to %s ...", report_path)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "total_turns": global_profile.total_turns,
            "split": "train",
        },
        "headline_scores": {
            k: v.score for k, v in global_profile.dimensions.items()
        },
        "dimensions": global_profile.to_dict()["dimensions"],
        "style_comparisons": {
            name: {
                k: v.score for k, v in prof.dimensions.items()
            }
            for name, prof in style_profiles.items()
        },
        "execution_seconds": round(duration, 2),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 7. Print human-readable behavioral report
    print("\n" + "=" * 80)
    print("STAGE T021: BEHAVIORAL PATTERN & TRAIT EXTRACTION REPORT")
    print("=" * 80)
    print(f"Total Evaluated Turns: {global_profile.total_turns:,}")
    print("-" * 80)
    print("QUANTIFIED BEHAVIORAL DIMENSIONS (NORMALIZED [0.0 - 1.0]):")
    for dim_name, dim in global_profile.dimensions.items():
        bar_len = int(dim.score * 30)
        bar_str = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  * {dim_name.upper():26s}: [{bar_str}] {dim.score:5.2f} (conf: {dim.confidence:4.2f})")
    print("-" * 80)
    print("SUPPORTING EVIDENCE EXEMPLARS:")
    for dim_name in ["directness", "disagreement_style", "humor_playfulness", "technical_depth"]:
        dim = global_profile.dimensions[dim_name]
        print(f"\n  [{dim_name.upper()}] - {dim.description}")
        for ex in dim.supporting_exemplars[:2]:
            print(f"    - \"{ex['target_text']}\" (score: {ex['score']}) [Context: {ex['context_text']}]")
    print("=" * 80)
    print("\nPER-ARCHETYPE BEHAVIORAL BREAKDOWN (SCORES):")
    header = f"  {'Archetype':28s} | Direct | Verbos | Disagr | Humor  | Questn | Confid | Tech"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s_name, prof in style_profiles.items():
        d = prof.dimensions
        print(f"  {s_name:28s} |  {d['directness'].score:4.2f}  |  {d['verbosity'].score:4.2f}  |  {d['disagreement_style'].score:4.2f}  |  {d['humor_playfulness'].score:4.2f}  |  {d['question_initiative'].score:4.2f}  |  {d['confidence_assertion'].score:4.2f}  | {d['technical_depth'].score:4.2f}")
    print("=" * 80)
    logger.info("Stage T021 Behavioral Pattern Extraction complete!")


if __name__ == "__main__":
    main()
