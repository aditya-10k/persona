"""
Execution script for Stages T022 & T023: Global & Situational Style Profiling.
Synthesizes surface linguistics, syntax, code-switching, discourse functions,
and behavioral traits across all 2,750 training turns into production-grade
Global and Situational Style Profiles.
Saves artifacts to data/processed/style/.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Ensure root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.persona.style import StyleProfileBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_style_profiling")


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

    data_dir = PROJECT_ROOT / "data" / "processed"
    style_dir = data_dir / "style"
    style_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = data_dir / "train_pairs.jsonl"
    if not pairs_path.exists():
        logger.error("Required pairs file not found at %s", pairs_path)
        sys.exit(1)

    # 1. Load pairs
    logger.info("Loading training pairs from %s ...", pairs_path)
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    logger.info("Loaded %d training turns.", len(pairs))

    # 2. Build Global Style Profile (Stage T022)
    builder = StyleProfileBuilder()
    logger.info("Building Global Style Profile (Stage T022) across %d turns ...", len(pairs))
    start_time = time.perf_counter()
    global_style = builder.build_global_style(pairs, name="global_persona")
    global_duration = time.perf_counter() - start_time
    logger.info("Global style profiling completed in %.2f seconds.", global_duration)

    # 3. Build Situational Style Profiles (Stage T023)
    logger.info("Building Situational Style Profiles (Stage T023) across situations ...")
    start_time = time.perf_counter()
    situational_styles = builder.build_situational_styles(pairs, max_exemplars=5)
    sit_duration = time.perf_counter() - start_time
    logger.info("Situational style profiling completed in %.2f seconds.", sit_duration)

    # 4. Save Global Style Profile
    global_profile_path = style_dir / "global_style_profile.json"
    logger.info("Writing global style profile to %s ...", global_profile_path)
    with open(global_profile_path, "w", encoding="utf-8") as f:
        json.dump(global_style.to_dict(), f, indent=2, ensure_ascii=False)

    # 5. Save Situational Style Profiles
    situational_profiles_path = style_dir / "situational_style_profiles.json"
    logger.info("Writing situational style profiles to %s ...", situational_profiles_path)
    sit_dict = {sit: prof.to_dict() for sit, prof in situational_styles.items()}
    with open(situational_profiles_path, "w", encoding="utf-8") as f:
        json.dump(sit_dict, f, indent=2, ensure_ascii=False)

    # 6. Save Style Summary Report
    report_path = style_dir / "style_report.json"
    logger.info("Writing style summary report to %s ...", report_path)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "total_turns": global_style.total_turns,
            "split": "train",
        },
        "global_style": global_style.to_dict(),
        "situations_covered": list(situational_styles.keys()),
        "situational_prevalences": {
            sit: prof.situation_prevalence for sit, prof in situational_styles.items()
        },
        "execution_seconds": {
            "global_style": round(global_duration, 2),
            "situational_styles": round(sit_duration, 2),
            "total": round(global_duration + sit_duration, 2),
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 7. Print human-readable report
    print("\n" + "=" * 80)
    print("STAGE T022: GLOBAL PERSONA STYLE PROFILE REPORT")
    print("=" * 80)
    print(f"Total Evaluated Turns: {global_style.total_turns:,}")
    print("-" * 80)
    print("SYNTHESIZED CORE BEHAVIORAL DIMENSIONS:")
    for dim, val in global_style.core_dimensions.items():
        bar_len = int(val * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        print(f"  * {dim.upper():26s}: [{bar}] {val:5.2f}")
    print("-" * 80)
    print("SURFACE LINGUISTIC HARD CONSTRAINTS:")
    print(f"  * Zero Terminal Punctuation : {global_style.surface_constraints['zero_terminal_punctuation_ratio'] * 100:.1f}%")
    print(f"  * All-Lowercase Casing      : {global_style.surface_constraints['all_lowercase_ratio'] * 100:.1f}%")
    print(f"  * Multi-Bubble Burst Ratio  : {global_style.surface_constraints['multi_bubble_burst_ratio'] * 100:.1f}% (mean {global_style.surface_constraints['mean_bubbles_per_turn']:.2f} msgs/turn)")
    print(f"  * Emoji Turn Prevalence     : {global_style.surface_constraints['emoji_turns_ratio'] * 100:.1f}% (top: {', '.join([e['emoji'] for e in global_style.surface_constraints['top_emojis']])})")
    print("-" * 80)
    print("SYNTACTIC & LANGUAGE MECHANICS:")
    print(f"  * Verbless Fragments Ratio  : {global_style.syntactic_constraints['verbless_fragment_ratio'] * 100:.1f}%")
    print(f"  * Complex Subordination     : {global_style.syntactic_constraints['complex_subordinate_ratio'] * 100:.1f}%")
    print(f"  * English vs Hindi Tokens   : {global_style.language_constraints['english_token_ratio'] * 100:.1f}% Eng / {global_style.language_constraints['hindi_token_ratio'] * 100:.1f}% Hin")
    print(f"  * Code-Switched Turn Ratio  : {global_style.language_constraints['code_switched_turns_ratio'] * 100:.1f}% (mean {global_style.language_constraints['mean_switches_per_turn']:.2f} switches/turn)")
    print("=" * 80)
    print("\nSTAGE T023: SITUATIONAL STYLE MODULATIONS")
    print("=" * 80)
    header = f"  {'Situation':22s} | Prev % | Direct | Verbos | Confid | Disagr | Humor  | Tech   | Eng %"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for sit_key, prof in situational_styles.items():
        prev = prof.situation_prevalence * 100
        cd = prof.core_dimensions
        eng_tok = prof.language_constraints["english_token_ratio"] * 100
        print(f"  {sit_key:22s} | {prev:5.1f}% |  {cd['directness']:4.2f}  |  {cd['verbosity']:4.2f}  |  {cd['confidence']:4.2f}  |  {cd['disagreement_style']:4.2f}  |  {cd['humor']:4.2f}  |  {cd['technical_depth']:4.2f}  | {eng_tok:4.1f}%")
    print("=" * 80)
    logger.info("Stages T022 & T023 Style Profiling complete!")


if __name__ == "__main__":
    main()
