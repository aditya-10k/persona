"""
Execution script for Stage T019: Discourse Analysis.
Analyzes 2,750 training turns to produce Global Discourse Profiles,
context-to-response transition probabilities P(Response Act | Context Act),
interpersonal positioning, multi-act co-occurrences, and per-archetype discourse breakdowns.
Saves artifacts to data/processed/discourse/.
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

from src.nlp.discourse import DiscourseAct, DiscourseAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_discourse_analysis")

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
    clustering_dir = data_dir / "clustering"
    discourse_dir = data_dir / "discourse"
    discourse_dir.mkdir(parents=True, exist_ok=True)

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

    # 2. Extract Global Discourse Profile
    analyzer = DiscourseAnalyzer()
    logger.info("Computing Global Discourse Profile across %d turns ...", len(pairs))
    start_time = time.perf_counter()
    global_profile = analyzer.fit(pairs, name="global_corpus")
    duration = time.perf_counter() - start_time
    logger.info("Global discourse profiling completed in %.2f seconds.", duration)

    # 3. Extract Per-Archetype Discourse Profiles
    logger.info("Computing per-archetype discourse breakdowns for the 6 styles ...")
    style_profiles = analyzer.analyze_by_style(pairs, style_labels, style_names=STYLE_NAMES)

    # 4. Save Global Profile
    global_profile_path = discourse_dir / "global_discourse_profile.json"
    logger.info("Writing global profile to %s ...", global_profile_path)
    with open(global_profile_path, "w", encoding="utf-8") as f:
        json.dump(global_profile.to_dict(), f, indent=2, ensure_ascii=False)

    # 5. Save Per-Style Profiles
    style_profiles_path = discourse_dir / "style_discourse_profiles.json"
    logger.info("Writing style profiles to %s ...", style_profiles_path)
    style_profiles_dict = {name: prof.to_dict() for name, prof in style_profiles.items()}
    with open(style_profiles_path, "w", encoding="utf-8") as f:
        json.dump(style_profiles_dict, f, indent=2, ensure_ascii=False)

    # 6. Save Comprehensive Discourse Report
    report_path = discourse_dir / "discourse_report.json"
    logger.info("Writing discourse summary report to %s ...", report_path)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "total_turns": global_profile.total_turns,
            "split": "train",
        },
        "act_distribution": global_profile.act_distribution,
        "act_counts": global_profile.act_counts,
        "top_markers_by_act": global_profile.top_markers_by_act,
        "context_response_transitions": global_profile.context_response_transitions,
        "style_comparisons": {
            name: {
                "total_turns": prof.total_turns,
                "act_distribution": prof.act_distribution,
                "top_act": max(prof.act_distribution.items(), key=lambda x: x[1])[0],
            }
            for name, prof in style_profiles.items()
        },
        "execution_seconds": round(duration, 2),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 7. Print human-readable discourse summary
    print("\n" + "=" * 80)
    print("STAGE T019: DISCOURSE ANALYSIS & COMMUNICATIVE ACT REPORT")
    print("=" * 80)
    print(f"Total Evaluated Turns:  {global_profile.total_turns:,}")
    print("-" * 80)
    print("GLOBAL DISCOURSE ACT DISTRIBUTION (PRIMARY COMMUNICATIVE MOVES):")
    for act in DiscourseAct:
        ratio = global_profile.act_distribution[act.value] * 100
        count = global_profile.act_counts.get(act.value, 0)
        top_m = [m["marker"] for m in global_profile.top_markers_by_act.get(act.value, [])[:4]]
        markers_str = f" [markers: {', '.join(top_m)}]" if top_m else ""
        print(f"  * {act.value.upper():12s}: {ratio:5.1f}% ({count:4d}){markers_str}")
    print("-" * 80)
    print("EMPIRICAL CONTEXT -> RESPONSE TRANSITIONS P(Response | Context):")
    for ctx_act in [DiscourseAct.QUESTION, DiscourseAct.DISAGREE, DiscourseAct.AGREE, DiscourseAct.HUMOR]:
        row = global_profile.context_response_transitions.get(ctx_act.value, {})
        top_responses = sorted(row.items(), key=lambda x: x[1], reverse=True)[:3]
        resp_str = ", ".join([f"{r.upper()} ({p * 100:.1f}%)" for r, p in top_responses if p > 0])
        print(f"  When Context is {ctx_act.value.upper():10s} -> Response: {resp_str}")
    print("=" * 80)
    print("\nPER-ARCHETYPE DISCOURSE BREAKDOWN:")
    for s_name, prof in style_profiles.items():
        agr = prof.act_distribution[DiscourseAct.AGREE.value] * 100
        dis = prof.act_distribution[DiscourseAct.DISAGREE.value] * 100
        qst = prof.act_distribution[DiscourseAct.QUESTION.value] * 100
        exp = prof.act_distribution[DiscourseAct.EXPLAIN.value] * 100
        hum = prof.act_distribution[DiscourseAct.HUMOR.value] * 100
        inf = prof.act_distribution[DiscourseAct.INFORMATIVE.value] * 100
        adv = prof.act_distribution[DiscourseAct.ADVICE.value] * 100
        hdg = prof.act_distribution[DiscourseAct.HEDGE.value] * 100
        print(f"  [{s_name:28s}] Agr: {agr:4.1f}% | Dis: {dis:4.1f}% | Qst: {qst:4.1f}% | Exp: {exp:4.1f}% | Hum: {hum:4.1f}% | Inf: {inf:4.1f}% | Adv: {adv:4.1f}% | Hdg: {hdg:4.1f}%")
    print("=" * 80)
    logger.info("Stage T019 Discourse Analysis complete!")


if __name__ == "__main__":
    main()
