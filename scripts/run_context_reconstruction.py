import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.segmentation import reconstruct_context_dataset

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    input_file = "data/processed/conversations.jsonl"
    output_file = "data/processed/context_pairs.jsonl"
    report_file = "data/processed/context_pairs_report.json"

    print(f"Starting T008 Context Reconstruction over {input_file} (max_context_turns=3)...")
    start_t = time.time()
    report = reconstruct_context_dataset(input_file, output_file, report_file, max_context_turns=3, target_speaker="You")
    elapsed = time.time() - start_t

    print(f"Completed in {elapsed:.2f}s")
    print(f"Total context pairs: {report['total_context_pairs']}")
    print(f"Initiation pairs: {report['initiation_pairs']}")
    print(f"Reply pairs: {report['reply_pairs']}")
    print("Context depth distribution:", report["context_depth_distribution"])
    print("Response latency stats:", report["response_latency_stats"])
    print("Top files by pairs:", list(report["pairs_by_source_file"].items())[:5])

if __name__ == "__main__":
    main()
