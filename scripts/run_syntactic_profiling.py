"""
Execution script for Stage T017: Syntactic Profiler.
Analyzes 2,750 training turns to produce Global Syntactic Profiles and
per-archetype syntactic breakdowns across clause structures, speech acts,
negation mechanics, conditionals, and pronoun orientations.
Saves artifacts to data/processed/syntax/.
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

from src.nlp.syntax import SyntacticProfiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_syntactic_profiling")

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
    syntax_dir = data_dir / "syntax"
    syntax_dir.mkdir(parents=True, exist_ok=True)

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

    # 2. Extract Global Syntactic Profile
    profiler = SyntacticProfiler()
    logger.info("Computing Global Syntactic Profile across %d turns ...", len(pairs))
    start_time = time.perf_counter()
    global_profile = profiler.fit(pairs, name="global_corpus")
    duration = time.perf_counter() - start_time
    logger.info("Global syntactic profiling completed in %.2f seconds.", duration)

    # 3. Extract Per-Archetype Syntactic Profiles
    logger.info("Computing per-archetype syntactic breakdowns for the 6 styles ...")
    style_profiles = profiler.analyze_by_style(pairs, style_labels, style_names=STYLE_NAMES)

    # 4. Save Global Profile
    global_profile_path = syntax_dir / "global_syntactic_profile.json"
    logger.info("Writing global profile to %s ...", global_profile_path)
    with open(global_profile_path, "w", encoding="utf-8") as f:
        json.dump(global_profile.to_dict(), f, indent=2, ensure_ascii=False)

    # 5. Save Per-Style Profiles
    style_profiles_path = syntax_dir / "style_syntactic_profiles.json"
    logger.info("Writing style profiles to %s ...", style_profiles_path)
    style_profiles_dict = {name: prof.to_dict() for name, prof in style_profiles.items()}
    with open(style_profiles_path, "w", encoding="utf-8") as f:
        json.dump(style_profiles_dict, f, indent=2, ensure_ascii=False)

    # 6. Save Comprehensive Syntax Report
    report_path = syntax_dir / "syntax_report.json"
    logger.info("Writing syntax summary report to %s ...", report_path)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "total_turns": global_profile.total_turns,
            "split": "train",
        },
        "headline_metrics": {
            "verbless_fragment_ratio": global_profile.clause_structure["verbless_fragment_ratio"],
            "simple_clause_ratio": global_profile.clause_structure["simple_clause_ratio"],
            "compound_coordinate_ratio": global_profile.clause_structure["compound_coordinate_ratio"],
            "complex_subordinate_ratio": global_profile.clause_structure["complex_subordinate_ratio"],
            "imperative_directive_ratio": global_profile.speech_acts["imperative_directive_ratio"],
            "inquisitive_question_ratio": global_profile.speech_acts["inquisitive_question_ratio"],
            "negation_turns_ratio": global_profile.negation_and_conditionals["negation_turns_ratio"],
            "conditional_turns_ratio": global_profile.negation_and_conditionals["conditional_turns_ratio"],
            "first_person_ratio": global_profile.pronoun_orientation["first_person_ratio"],
            "second_person_ratio": global_profile.pronoun_orientation["second_person_ratio"],
            "self_to_other_ratio": global_profile.pronoun_orientation["self_to_other_ratio"],
        },
        "style_comparisons": {
            name: {
                "total_turns": prof.total_turns,
                "verbless_fragment_ratio": prof.clause_structure["verbless_fragment_ratio"],
                "imperative_ratio": prof.speech_acts["imperative_directive_ratio"],
                "inquisitive_ratio": prof.speech_acts["inquisitive_question_ratio"],
                "negation_ratio": prof.negation_and_conditionals["negation_turns_ratio"],
                "first_person_ratio": prof.pronoun_orientation["first_person_ratio"],
                "second_person_ratio": prof.pronoun_orientation["second_person_ratio"],
            }
            for name, prof in style_profiles.items()
        },
        "execution_seconds": round(duration, 2),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 7. Print human-readable syntactic summary
    print("\n" + "=" * 80)
    print("STAGE T017: SYNTACTIC FINGERPRINT REPORT")
    print("=" * 80)
    print(f"Total Evaluated Turns: {global_profile.total_turns:,}")
    print("-" * 80)
    print("CLAUSE STRUCTURE & COMPLEXITY:")
    print(f"  * Verbless Fragments (≤2 words / pings): {global_profile.clause_structure['verbless_fragment_ratio'] * 100:.1f}%")
    print(f"  * Simple Independent Clauses:            {global_profile.clause_structure['simple_clause_ratio'] * 100:.1f}%")
    print(f"  * Compound Coordinate Clauses (and/aur):  {global_profile.clause_structure['compound_coordinate_ratio'] * 100:.1f}%")
    print(f"  * Complex Subordinate Clauses (kyuki/if): {global_profile.clause_structure['complex_subordinate_ratio'] * 100:.1f}%")
    print("-" * 80)
    print("SPEECH ACTS & SENTENCE FUNCTIONS:")
    print(f"  * Inquisitives / Questions (Wh / '?'):   {global_profile.speech_acts['inquisitive_question_ratio'] * 100:.1f}%")
    print(f"  * Imperatives / Directives (Commands):   {global_profile.speech_acts['imperative_directive_ratio'] * 100:.1f}%")
    print(f"  * Declarative Statements:                {global_profile.speech_acts['declarative_ratio'] * 100:.1f}%")
    print("-" * 80)
    print("NEGATION & CONTINGENCY:")
    print(f"  * Negation Turns (nahi/mat/no):          {global_profile.negation_and_conditionals['negation_turns_ratio'] * 100:.1f}%")
    print(f"  * Conditional Hypotheses (agar/if/warna): {global_profile.negation_and_conditionals['conditional_turns_ratio'] * 100:.1f}%")
    print("-" * 80)
    print("PRONOUN ORIENTATION:")
    print(f"  * 1st Person Turns (me/mera/i/we):       {global_profile.pronoun_orientation['first_person_ratio'] * 100:.1f}%")
    print(f"  * 2nd Person Turns (tu/tera/you):        {global_profile.pronoun_orientation['second_person_ratio'] * 100:.1f}%")
    print(f"  * 3rd Person Turns (wo/uske/he/they):    {global_profile.pronoun_orientation['third_person_ratio'] * 100:.1f}%")
    print(f"  * Self-to-Other Orientation Ratio:       {global_profile.pronoun_orientation['self_to_other_ratio']}x")
    print("=" * 80)
    print("\nPER-ARCHETYPE SYNTACTIC BREAKDOWN:")
    for s_name, prof in style_profiles.items():
        frag_pct = prof.clause_structure["verbless_fragment_ratio"] * 100
        imp_pct = prof.speech_acts["imperative_directive_ratio"] * 100
        inq_pct = prof.speech_acts["inquisitive_question_ratio"] * 100
        neg_pct = prof.negation_and_conditionals["negation_turns_ratio"] * 100
        p1_pct = prof.pronoun_orientation["first_person_ratio"] * 100
        p2_pct = prof.pronoun_orientation["second_person_ratio"] * 100
        print(f"  [{s_name:28s}] Fragment: {frag_pct:4.1f}% | Imperative: {imp_pct:4.1f}% | Inquisitive: {inq_pct:4.1f}% | Negation: {neg_pct:4.1f}% | P1: {p1_pct:4.1f}% | P2: {p2_pct:4.1f}%")
    print("=" * 80)
    logger.info("Stage T017 Syntactic Profiling complete!")


if __name__ == "__main__":
    main()
