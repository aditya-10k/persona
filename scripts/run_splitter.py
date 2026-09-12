import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import create_dataset_splits

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    pairs_file = "data/processed/sampled_pairs.jsonl"
    convs_file = "data/processed/conversations.jsonl"
    out_dir = "data/processed"
    report_file = "data/processed/splits_report.json"

    print(f"Starting T011 Dataset Splitting over {pairs_file}...")
    start_t = time.time()
    report = create_dataset_splits(
        pairs_file,
        convs_file,
        out_dir,
        report_file,
        train_ratio=0.70,
        dev_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )
    elapsed = time.time() - start_t

    print(f"Completed in {elapsed:.2f}s")
    print(f"Zero leakage verified: {report['zero_leakage_verified']}")
    print("Conversations split:", report["conversations_split"])
    print("Pairs split:", report["pairs_split"])
    print("Length breakdown per split:", report["length_bin_distribution_by_split"])

if __name__ == "__main__":
    main()
