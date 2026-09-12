import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import sample_context_pairs_dataset

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    input_file = "data/processed/filtered_pairs.jsonl"
    output_file = "data/processed/sampled_pairs.jsonl"
    report_file = "data/processed/sampling_report.json"

    print(f"Starting T010 Stratified Sampling over {input_file} (max_per_source=500, seed=42)...")
    start_t = time.time()
    report = sample_context_pairs_dataset(input_file, output_file, report_file, max_per_source=500, random_seed=42)
    elapsed = time.time() - start_t

    print(f"Completed in {elapsed:.2f}s")
    print(f"Total pairs before: {report['total_pairs_before']}")
    print(f"Total pairs after: {report['total_pairs_after']} ({report['sampling_ratio_percentage']}%)")
    print("Length distribution (after):", report["length_bin_distribution"])
    print("Initiation retention:", report["initiations"])
    print("\nSource distribution comparison:")
    for src, stats in report["source_distribution"].items():
        print(f"  {src:<40}: before={stats['before']:<5} ({stats['before_pct']:>4.1f}%) -> after={stats['after']:<4} ({stats['after_pct']:>4.1f}%)")

if __name__ == "__main__":
    main()
