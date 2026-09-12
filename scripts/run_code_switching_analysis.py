"""
Execution script for Stage T018: Code-Switching & Language Analysis.
Analyzes 2,750 training turns to produce Global Language Profiles and
per-archetype language breakdowns across English vs Romanized Hindi distributions,
turn classification modes, code-switching transition dynamics, and borrowed English loanwords.
Saves artifacts to data/processed/language/.
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

from src.nlp.code_switching import CodeSwitchingAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_code_switching_analysis")

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
    lang_dir = data_dir / "language"
    lang_dir.mkdir(parents=True, exist_ok=True)

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

    # 2. Extract Global Language Profile
    profiler = CodeSwitchingAnalyzer()
    logger.info("Computing Global Language Profile across %d turns ...", len(pairs))
    start_time = time.perf_counter()
    global_profile = profiler.fit(pairs, name="global_corpus")
    duration = time.perf_counter() - start_time
    logger.info("Global code-switching profiling completed in %.2f seconds.", duration)

    # 3. Extract Per-Archetype Language Profiles
    logger.info("Computing per-archetype language breakdowns for the 6 styles ...")
    style_profiles = profiler.analyze_by_style(pairs, style_labels, style_names=STYLE_NAMES)

    # 4. Save Global Profile
    global_profile_path = lang_dir / "global_language_profile.json"
    logger.info("Writing global profile to %s ...", global_profile_path)
    with open(global_profile_path, "w", encoding="utf-8") as f:
        json.dump(global_profile.to_dict(), f, indent=2, ensure_ascii=False)

    # 5. Save Per-Style Profiles
    style_profiles_path = lang_dir / "style_language_profiles.json"
    logger.info("Writing style profiles to %s ...", style_profiles_path)
    style_profiles_dict = {name: prof.to_dict() for name, prof in style_profiles.items()}
    with open(style_profiles_path, "w", encoding="utf-8") as f:
        json.dump(style_profiles_dict, f, indent=2, ensure_ascii=False)

    # 6. Save Comprehensive Code-Switching Report
    report_path = lang_dir / "code_switching_report.json"
    logger.info("Writing code-switching summary report to %s ...", report_path)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "total_turns": global_profile.total_turns,
            "total_tokens": global_profile.total_tokens,
            "split": "train",
        },
        "headline_metrics": {
            "english_token_ratio": global_profile.token_distribution["english_token_ratio"],
            "hindi_token_ratio": global_profile.token_distribution["hindi_token_ratio"],
            "universal_token_ratio": global_profile.token_distribution["universal_token_ratio"],
            "pure_english_turns_ratio": global_profile.turn_classification["pure_english_ratio"],
            "pure_hindi_turns_ratio": global_profile.turn_classification["pure_hindi_ratio"],
            "code_switched_turns_ratio": global_profile.turn_classification["code_switched_ratio"],
            "neutral_turns_ratio": global_profile.turn_classification["neutral_ratio"],
            "mean_switches_per_turn": global_profile.switching_dynamics["mean_switches_per_turn"],
            "switched_turns_ratio": global_profile.switching_dynamics["switched_turns_ratio"],
            "max_switches_in_turn": global_profile.switching_dynamics["max_switches_in_turn"],
        },
        "style_comparisons": {
            name: {
                "total_turns": prof.total_turns,
                "english_token_ratio": prof.token_distribution["english_token_ratio"],
                "hindi_token_ratio": prof.token_distribution["hindi_token_ratio"],
                "pure_english_ratio": prof.turn_classification["pure_english_ratio"],
                "pure_hindi_ratio": prof.turn_classification["pure_hindi_ratio"],
                "code_switched_ratio": prof.turn_classification["code_switched_ratio"],
                "neutral_ratio": prof.turn_classification["neutral_ratio"],
                "mean_switches_per_turn": prof.switching_dynamics["mean_switches_per_turn"],
            }
            for name, prof in style_profiles.items()
        },
        "top_borrowed_english_words": global_profile.borrowed_english_words[:25],
        "execution_seconds": round(duration, 2),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 7. Print human-readable language summary
    print("\n" + "=" * 80)
    print("STAGE T018: CODE-SWITCHING & LANGUAGE PROFILE REPORT")
    print("=" * 80)
    print(f"Total Evaluated Turns:  {global_profile.total_turns:,}")
    print(f"Total Evaluated Tokens: {global_profile.total_tokens:,}")
    print("-" * 80)
    print("GLOBAL TOKEN LANGUAGE DISTRIBUTION:")
    print(f"  * English Tokens:   {global_profile.token_distribution['english_token_ratio'] * 100:.1f}% ({global_profile.token_distribution['total_english_tokens']:,})")
    print(f"  * Hindi Tokens:     {global_profile.token_distribution['hindi_token_ratio'] * 100:.1f}% ({global_profile.token_distribution['total_hindi_tokens']:,})")
    print(f"  * Universal Tokens: {global_profile.token_distribution['universal_token_ratio'] * 100:.1f}% ({global_profile.token_distribution['total_universal_tokens']:,})")
    print("-" * 80)
    print("TURN-LEVEL CONVERSATIONAL MODES:")
    print(f"  * Pure English Turns:    {global_profile.turn_classification['pure_english_ratio'] * 100:.1f}% ({global_profile.turn_classification['pure_english_turns']:,})")
    print(f"  * Pure Hindi Turns:      {global_profile.turn_classification['pure_hindi_ratio'] * 100:.1f}% ({global_profile.turn_classification['pure_hindi_turns']:,})")
    print(f"  * Code-Switched Hinglish: {global_profile.turn_classification['code_switched_ratio'] * 100:.1f}% ({global_profile.turn_classification['code_switched_turns']:,})")
    print(f"  * Neutral (Media/Emoji):  {global_profile.turn_classification['neutral_ratio'] * 100:.1f}% ({global_profile.turn_classification['neutral_turns']:,})")
    print("-" * 80)
    print("CODE-SWITCHING DYNAMICS:")
    print(f"  * Switched Turns Ratio:        {global_profile.switching_dynamics['switched_turns_ratio'] * 100:.1f}%")
    print(f"  * Mean Switches per Turn:      {global_profile.switching_dynamics['mean_switches_per_turn']:.2f}")
    print(f"  * Max Switches in Single Turn: {global_profile.switching_dynamics['max_switches_in_turn']}")
    print("-" * 80)
    print("TOP BORROWED ENGLISH LOANWORDS IN HINGLISH:")
    top_loanwords = [f"{b['word']} ({b['count']})" for b in global_profile.borrowed_english_words[:15]]
    print(f"  {', '.join(top_loanwords)}")
    print("=" * 80)
    print("\nPER-ARCHETYPE CODE-SWITCHING BREAKDOWN:")
    for s_name, prof in style_profiles.items():
        eng_tok = prof.token_distribution["english_token_ratio"] * 100
        hin_tok = prof.token_distribution["hindi_token_ratio"] * 100
        p_eng = prof.turn_classification["pure_english_ratio"] * 100
        p_hin = prof.turn_classification["pure_hindi_ratio"] * 100
        switched = prof.turn_classification["code_switched_ratio"] * 100
        sw_pts = prof.switching_dynamics["mean_switches_per_turn"]
        print(f"  [{s_name:28s}] Eng Tok: {eng_tok:4.1f}% | Hin Tok: {hin_tok:4.1f}% | Pure Eng: {p_eng:4.1f}% | Pure Hin: {p_hin:4.1f}% | Switched: {switched:4.1f}% | Switches/Turn: {sw_pts:3.1f}")
    print("=" * 80)
    logger.info("Stage T018 Code-Switching Analysis complete!")


if __name__ == "__main__":
    main()
