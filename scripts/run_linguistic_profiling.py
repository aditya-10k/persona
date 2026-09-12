"""
Execution script for Stage T016: Surface Linguistic Profiler.
Analyzes 2,750 training turns to produce the Global Linguistic Profile and
per-archetype linguistic breakdowns. Saves artifacts to data/processed/linguistics/.
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

from src.nlp.linguistics import LinguisticProfiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_linguistic_profiling")

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
    linguistics_dir = data_dir / "linguistics"
    linguistics_dir.mkdir(parents=True, exist_ok=True)

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

    # 2. Extract Global Linguistic Profile
    profiler = LinguisticProfiler()
    logger.info("Computing Global Linguistic Profile across %d turns ...", len(pairs))
    start_time = time.perf_counter()
    global_profile = profiler.fit(pairs, name="global_corpus")
    duration = time.perf_counter() - start_time
    logger.info("Global profiling completed in %.2f seconds.", duration)

    # 3. Extract Per-Archetype Linguistic Profiles
    logger.info("Computing per-archetype linguistic breakdowns for the 6 styles ...")
    style_profiles = profiler.analyze_by_style(pairs, style_labels, style_names=STYLE_NAMES)

    # 4. Save Global Profile
    global_profile_path = linguistics_dir / "global_linguistic_profile.json"
    logger.info("Writing global profile to %s ...", global_profile_path)
    with open(global_profile_path, "w", encoding="utf-8") as f:
        json.dump(global_profile.to_dict(), f, indent=2, ensure_ascii=False)

    # 5. Save Per-Style Profiles
    style_profiles_path = linguistics_dir / "style_linguistic_profiles.json"
    logger.info("Writing style profiles to %s ...", style_profiles_path)
    style_profiles_dict = {name: prof.to_dict() for name, prof in style_profiles.items()}
    with open(style_profiles_path, "w", encoding="utf-8") as f:
        json.dump(style_profiles_dict, f, indent=2, ensure_ascii=False)

    # 6. Save Comprehensive Linguistics Report
    report_path = linguistics_dir / "linguistics_report.json"
    logger.info("Writing linguistics summary report to %s ...", report_path)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "total_turns": global_profile.total_turns,
            "total_messages": global_profile.total_messages,
            "total_words": global_profile.total_words,
            "total_characters": global_profile.total_characters,
            "split": "train",
        },
        "headline_metrics": {
            "mean_words_per_turn": global_profile.words_per_turn["mean"],
            "median_words_per_turn": global_profile.words_per_turn["median"],
            "mean_messages_per_turn": global_profile.burstiness["mean_messages_per_turn"],
            "single_bubble_ratio": global_profile.burstiness["single_bubble_ratio"],
            "all_lowercase_ratio": global_profile.casing_distribution["all_lowercase_ratio"],
            "zero_terminal_punctuation_ratio": global_profile.punctuation_metrics["zero_terminal_punctuation_ratio"],
            "turns_with_emojis_ratio": global_profile.emoji_fingerprint["turns_with_emojis_ratio"],
            "type_token_ratio": global_profile.lexical_diversity["ttr"],
            "root_ttr_guiraud": global_profile.lexical_diversity["root_ttr_guiraud"],
            "hapax_legomena_ratio": global_profile.lexical_diversity["hapax_ratio"],
        },
        "style_comparisons": {
            name: {
                "total_turns": prof.total_turns,
                "mean_words": prof.words_per_turn["mean"],
                "all_lowercase_ratio": prof.casing_distribution["all_lowercase_ratio"],
                "zero_terminal_punctuation_ratio": prof.punctuation_metrics["zero_terminal_punctuation_ratio"],
                "turns_with_emojis_ratio": prof.emoji_fingerprint["turns_with_emojis_ratio"],
                "top_emojis": [e["emoji"] for e in prof.emoji_fingerprint["top_emojis"][:3]],
            }
            for name, prof in style_profiles.items()
        },
        "execution_seconds": round(duration, 2),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 7. Print human-readable linguistic summary
    print("\n" + "=" * 80)
    print("STAGE T016: SURFACE LINGUISTIC FINGERPRINT REPORT")
    print("=" * 80)
    print(f"Total Evaluated Turns:   {global_profile.total_turns:,}")
    print(f"Total Messages (Bubbles): {global_profile.total_messages:,} (Avg {global_profile.burstiness['mean_messages_per_turn']} msgs/turn)")
    print(f"Total Words (Tokens):     {global_profile.total_words:,} (Vocab Size: {global_profile.lexical_diversity['vocabulary_size']:,})")
    print(f"Type-Token Ratio (TTR):   {global_profile.lexical_diversity['ttr']:.4f} (Guiraud: {global_profile.lexical_diversity['root_ttr_guiraud']:.2f})")
    print(f"Hapax Legomena Ratio:     {global_profile.lexical_diversity['hapax_ratio'] * 100:.1f}% unique words used once")
    print("-" * 80)
    print("ORTHOGRAPHY & CASING:")
    print(f"  * All-Lowercase Turns:       {global_profile.casing_distribution['all_lowercase_ratio'] * 100:.1f}%")
    print(f"  * Initial Capital Turns:     {global_profile.casing_distribution['initial_capital_ratio'] * 100:.1f}%")
    print(f"  * All-Uppercase Shouting:    {global_profile.casing_distribution['all_uppercase_ratio'] * 100:.1f}%")
    print("-" * 80)
    print("PUNCTUATION MECHANICS:")
    print(f"  * Zero Terminal Punctuation: {global_profile.punctuation_metrics['zero_terminal_punctuation_ratio'] * 100:.1f}%")
    print(f"  * Question Turns:            {global_profile.punctuation_metrics['question_turns_ratio'] * 100:.1f}% (Multi '??': {global_profile.punctuation_metrics['multi_question_ratio'] * 100:.1f}%)")
    print(f"  * Ellipsis '...' Turns:      {global_profile.punctuation_metrics['ellipsis_turns_ratio'] * 100:.1f}%")
    print("-" * 80)
    print("EMOJI SIGNATURE:")
    print(f"  * Turns With Emojis:         {global_profile.emoji_fingerprint['turns_with_emojis_ratio'] * 100:.1f}%")
    print(f"  * Pure Emoji-Only Turns:     {global_profile.emoji_fingerprint['emoji_only_turns_ratio'] * 100:.1f}%")
    print(f"  * Emoji Bursts ('😂😂'):     {global_profile.emoji_fingerprint['emoji_burst_ratio'] * 100:.1f}% of emoji messages")
    top_em_str = ", ".join(f"{e['emoji']} ({e['count']})" for e in global_profile.emoji_fingerprint["top_emojis"][:8])
    print(f"  * Top Emojis:                {top_em_str}")
    print("-" * 80)
    print("BURSTINESS & CADENCE:")
    print(f"  * Single-Bubble Turns:       {global_profile.burstiness['single_bubble_ratio'] * 100:.1f}%")
    print(f"  * Double-Bubble Turns:       {global_profile.burstiness['double_bubble_ratio'] * 100:.1f}%")
    print(f"  * Multi-Bubble Turns (3+):   {global_profile.burstiness['multi_bubble_ratio'] * 100:.1f}% (Max burst: {global_profile.burstiness['max_burst_messages']})")
    print("=" * 80)
    print("\nPER-ARCHETYPE LINGUISTIC BREAKDOWN:")
    for s_name, prof in style_profiles.items():
        w_mean = prof.words_per_turn["mean"]
        low_pct = prof.casing_distribution["all_lowercase_ratio"] * 100
        zero_p = prof.punctuation_metrics["zero_terminal_punctuation_ratio"] * 100
        em_pct = prof.emoji_fingerprint["turns_with_emojis_ratio"] * 100
        em_list = " ".join(e["emoji"] for e in prof.emoji_fingerprint["top_emojis"][:3]) or "None"
        print(f"  [{s_name:28s}] Words: {w_mean:4.1f} | Lower: {low_pct:4.1f}% | NoPunct: {zero_p:4.1f}% | Emojis: {em_pct:4.1f}% ({em_list})")
    print("=" * 80)
    logger.info("Stage T016 Surface Linguistic Profiling complete!")


if __name__ == "__main__":
    main()
