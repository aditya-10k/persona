"""
Execution script for Stage T024: LLM-Assisted Structured Inference.
Loads global and situational style profiles, generates strict epistemic persona interpretations
(OBSERVED vs INFERRED vs UNKNOWN), and saves artifacts to data/processed/inference/.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Ensure root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.persona.inference import EpistemicStatus, StructuredInferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_structured_inference")


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

    data_dir = PROJECT_ROOT / "data" / "processed"
    style_dir = data_dir / "style"
    inf_dir = data_dir / "inference"
    inf_dir.mkdir(parents=True, exist_ok=True)

    global_style_path = style_dir / "global_style_profile.json"
    sit_styles_path = style_dir / "situational_style_profiles.json"

    if not global_style_path.exists() or not sit_styles_path.exists():
        logger.error("Required style profile files not found in %s", style_dir)
        sys.exit(1)

    logger.info("Loading style profiles from %s ...", style_dir)
    with open(global_style_path, "r", encoding="utf-8") as f:
        global_style = json.load(f)

    with open(sit_styles_path, "r", encoding="utf-8") as f:
        sit_styles = json.load(f)

    # 1. Run Structured Inference
    engine = StructuredInferenceEngine()
    logger.info("Synthesizing structured persona inference with strict epistemic boundaries ...")
    start_time = time.perf_counter()
    inference_result = engine.infer_offline(global_style, sit_styles)
    duration = time.perf_counter() - start_time
    logger.info("Structured inference completed in %.2f seconds.", duration)

    # 2. Save Inference Result
    inference_path = inf_dir / "structured_persona_inference.json"
    logger.info("Writing structured inference to %s ...", inference_path)
    with open(inference_path, "w", encoding="utf-8") as f:
        json.dump(inference_result.to_dict(), f, indent=2, ensure_ascii=False)

    # 3. Generate and Save LLM Prompt Template for Downstream Verification
    prompt_template = engine.generate_prompt(global_style, sit_styles, exemplars=[])
    prompt_path = inf_dir / "llm_inference_prompt_template.txt"
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt_template)

    # 4. Print human-readable inference report
    print("\n" + "=" * 80)
    print("STAGE T024: STRUCTURED PERSONA INFERENCE REPORT")
    print("=" * 80)
    print(f"PERSONA ARCHETYPE: {inference_result.persona_archetype}")
    print(f"CORE SUMMARY:     {inference_result.core_summary}")
    print("-" * 80)
    print("1. OBSERVED TRAITS (EMPIRICALLY BACKED BY HARD NUMBERS):")
    for item in inference_result.observed_traits:
        print(f"  * [{item.trait:32s}] (conf: {item.confidence:.2f})")
        print(f"    {item.description}")
    print("-" * 80)
    print("2. INFERRED TRAITS (CONSERVATIVE GENERALIZATIONS):")
    for item in inference_result.inferred_traits:
        print(f"  * [{item.trait:32s}] (conf: {item.confidence:.2f})")
        print(f"    {item.description}")
    print("-" * 80)
    print("3. UNKNOWN / QUARANTINED DOMAINS (STRICTLY PROHIBITED FROM HALLUCINATION):")
    for item in inference_result.unknown_aspects:
        print(f"  * [{item.trait:32s}] (conf: {item.confidence:.2f})")
        print(f"    {item.description}")
    print("=" * 80)
    logger.info("Stage T024 Structured Inference complete!")


if __name__ == "__main__":
    main()
