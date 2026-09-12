import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import filter_context_pairs_dataset

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    input_file = "data/processed/context_pairs.jsonl"
    output_file = "data/processed/filtered_pairs.jsonl"
    report_file = "data/processed/quality_filtering_report.json"

    print(f"Starting T009 Quality Filtering over {input_file}...")
    start_t = time.time()
    report = filter_context_pairs_dataset(input_file, output_file, report_file)
    elapsed = time.time() - start_t

    print(f"Completed in {elapsed:.2f}s")
    print(f"Total processed: {report['total_pairs_processed']}")
    print(f"Total passed: {report['total_pairs_passed']} ({report['pass_rate_percentage']}%)")
    print(f"Total filtered: {report['total_pairs_filtered']}")
    print("Filter breakdown:", report["filter_reasons_breakdown"])
    print("Passed initiations:", report["passed_pairs_metrics"]["initiation_pairs"])
    print("Passed replies:", report["passed_pairs_metrics"]["reply_pairs"])
    print("Passed by depth:", report["passed_pairs_metrics"]["by_context_depth"])

if __name__ == "__main__":
    main()
