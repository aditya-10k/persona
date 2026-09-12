"""
Execution script for Stage T025: Evidence Aggregation.
Combines statistical profiles, situational profiles, behavioral dimensions,
and structured inference into the Master Evidence Registry.
Saves artifacts to data/processed/evidence/.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Ensure root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.persona.evidence import EvidenceAggregator
from src.persona.inference import StructuredInferenceEngine
from src.persona.style import GlobalStyleProfile, SituationalStyleProfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_evidence_aggregation")


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

    data_dir = PROJECT_ROOT / "data" / "processed"
    style_dir = data_dir / "style"
    inf_dir = data_dir / "inference"
    evidence_dir = data_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    global_style_path = style_dir / "global_style_profile.json"
    sit_styles_path = style_dir / "situational_style_profiles.json"
    inference_path = inf_dir / "structured_persona_inference.json"

    if not global_style_path.exists() or not sit_styles_path.exists():
        logger.error("Required style files missing in %s", style_dir)
        sys.exit(1)

    logger.info("Loading global and situational style profiles ...")
    with open(global_style_path, "r", encoding="utf-8") as f:
        g_raw = json.load(f)
        global_style = GlobalStyleProfile(
            total_turns=g_raw["total_turns"],
            core_dimensions=g_raw["core_dimensions"],
            surface_constraints=g_raw["surface_constraints"],
            syntactic_constraints=g_raw["syntactic_constraints"],
            language_constraints=g_raw["language_constraints"],
            discourse_moves=g_raw["discourse_moves"],
            metadata=g_raw.get("metadata", {}),
        )

    with open(sit_styles_path, "r", encoding="utf-8") as f:
        s_raw = json.load(f)
        situational_styles = {
            k: SituationalStyleProfile(
                situation=v["situation"],
                description=v["description"],
                total_turns=v["total_turns"],
                situation_prevalence=v["situation_prevalence"],
                core_dimensions=v["core_dimensions"],
                surface_constraints=v["surface_constraints"],
                language_constraints=v["language_constraints"],
                discourse_moves=v["discourse_moves"],
                representative_exemplars=v.get("representative_exemplars", []),
                metadata=v.get("metadata", {}),
            )
            for k, v in s_raw.items()
        }

    # Load or generate inference result
    engine = StructuredInferenceEngine()
    inference_result = engine.infer_offline(g_raw, s_raw)

    # Aggregate Evidence
    aggregator = EvidenceAggregator()
    logger.info("Aggregating multi-source evidence into Master Evidence Registry ...")
    start_time = time.perf_counter()
    registry = aggregator.aggregate(
        global_style=global_style,
        situational_styles=situational_styles,
        inference_result=inference_result,
    )
    duration = time.perf_counter() - start_time
    logger.info("Evidence aggregation completed in %.2f seconds.", duration)

    # Save Registry
    registry_path = evidence_dir / "evidence_registry.json"
    logger.info("Writing evidence registry to %s ...", registry_path)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry.to_dict(), f, indent=2, ensure_ascii=False)

    # Print human-readable audit
    print("\n" + "=" * 80)
    print("STAGE T025: MASTER EVIDENCE REGISTRY & AUDIT REPORT")
    print("=" * 80)
    print(f"Total Evaluated Turns:   {registry.total_evaluated_turns:,}")
    print(f"Total Evidence Items:    {registry.total_evidence_items}")
    print(f"Epistemic Breakdown:     Observed: {registry.epistemic_breakdown.get('observed', 0)} | Inferred: {registry.epistemic_breakdown.get('inferred', 0)} | Unknown: {registry.epistemic_breakdown.get('unknown', 0)}")
    print("-" * 80)
    header = f"  {'Trait / Rule':32s} | Status   | Score | Conf  | Evidence Count"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key, item in registry.items.items():
        print(f"  {item.trait:32s} | {item.epistemic_status:8s} | {item.score:5.2f} | {item.confidence:5.2f} | {item.evidence_count:6d} turns")
    print("=" * 80)
    logger.info("Stage T025 Evidence Aggregation complete!")


if __name__ == "__main__":
    main()
