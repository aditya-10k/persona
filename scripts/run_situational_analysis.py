"""
Execution script for Stage T020: Situational Classification.
Classifies all 2,750 training turns into situational environments:
Technical Collab, Casual Banter, Conflict / Friction, Career / Academic, Acknowledgement, Advice / Probing.
Saves artifacts to data/processed/situations/.
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

from src.nlp.situations import SituationalCategory, SituationalClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_situational_analysis")

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
    sit_dir = data_dir / "situations"
    sit_dir.mkdir(parents=True, exist_ok=True)

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

    # 2. Extract Global Situational Profile
    classifier = SituationalClassifier()
    logger.info("Computing Global Situational Profile across %d turns ...", len(pairs))
    start_time = time.perf_counter()
    global_profile = classifier.fit(pairs, name="global_corpus")
    duration = time.perf_counter() - start_time
    logger.info("Global situational classification completed in %.2f seconds.", duration)

    # 3. Extract Per-Archetype Situational Profiles
    logger.info("Computing per-archetype situational breakdowns for the 6 styles ...")
    style_profiles = classifier.analyze_by_style(pairs, style_labels, style_names=STYLE_NAMES)

    # 4. Save Global Profile
    global_profile_path = sit_dir / "global_situational_profile.json"
    logger.info("Writing global profile to %s ...", global_profile_path)
    with open(global_profile_path, "w", encoding="utf-8") as f:
        json.dump(global_profile.to_dict(), f, indent=2, ensure_ascii=False)

    # 5. Save Per-Style Profiles
    style_profiles_path = sit_dir / "style_situational_profiles.json"
    logger.info("Writing style profiles to %s ...", style_profiles_path)
    style_profiles_dict = {name: prof.to_dict() for name, prof in style_profiles.items()}
    with open(style_profiles_path, "w", encoding="utf-8") as f:
        json.dump(style_profiles_dict, f, indent=2, ensure_ascii=False)

    # 6. Save Comprehensive Situational Report
    report_path = sit_dir / "situational_report.json"
    logger.info("Writing situational summary report to %s ...", report_path)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "total_turns": global_profile.total_turns,
            "split": "train",
        },
        "situation_distribution": global_profile.situation_distribution,
        "situation_counts": global_profile.situation_counts,
        "top_cues_by_situation": global_profile.top_cues_by_situation,
        "style_comparisons": {
            name: {
                "total_turns": prof.total_turns,
                "situation_distribution": prof.situation_distribution,
                "top_situation": max(prof.situation_distribution.items(), key=lambda x: x[1])[0],
            }
            for name, prof in style_profiles.items()
        },
        "execution_seconds": round(duration, 2),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 7. Print human-readable summary
    print("\n" + "=" * 80)
    print("STAGE T020: SITUATIONAL CLASSIFICATION REPORT")
    print("=" * 80)
    print(f"Total Evaluated Turns: {global_profile.total_turns:,}")
    print("-" * 80)
    print("GLOBAL SITUATION DISTRIBUTION:")
    for sit in SituationalCategory:
        ratio = global_profile.situation_distribution[sit.value] * 100
        count = global_profile.situation_counts.get(sit.value, 0)
        top_c = [c["cue"] for c in global_profile.top_cues_by_situation.get(sit.value, [])[:4]]
        cues_str = f" [cues: {', '.join(top_c)}]" if top_c else ""
        print(f"  * {sit.value.upper():20s}: {ratio:5.1f}% ({count:4d}){cues_str}")
    print("=" * 80)
    print("\nPER-ARCHETYPE SITUATIONAL BREAKDOWN:")
    for s_name, prof in style_profiles.items():
        tech = prof.situation_distribution[SituationalCategory.TECHNICAL_COLLAB.value] * 100
        cas = prof.situation_distribution[SituationalCategory.CASUAL_BANTER.value] * 100
        con = prof.situation_distribution[SituationalCategory.CONFLICT_FRICTION.value] * 100
        car = prof.situation_distribution[SituationalCategory.CAREER_ACADEMIC.value] * 100
        ack = prof.situation_distribution[SituationalCategory.ACKNOWLEDGEMENT.value] * 100
        adv = prof.situation_distribution[SituationalCategory.ADVICE_PROBING.value] * 100
        print(f"  [{s_name:28s}] Tech: {tech:4.1f}% | Casual: {cas:4.1f}% | Confl: {con:4.1f}% | Career: {car:4.1f}% | Ack: {ack:4.1f}% | Advice: {adv:4.1f}%")
    print("=" * 80)
    logger.info("Stage T020 Situational Classification complete!")


if __name__ == "__main__":
    main()
