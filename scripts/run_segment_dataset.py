import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.segmentation import segment_dataset

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    input_file = "data/processed/cleaned_messages.jsonl"
    output_file = "data/processed/conversations.jsonl"
    report_file = "data/processed/segmentation_report.json"

    print(f"Starting T007 Conversation Segmentation over {input_file} (gap_hours=4.0)...")
    start_t = time.time()
    report = segment_dataset(input_file, output_file, report_file, gap_hours=4.0)
    elapsed = time.time() - start_t

    print(f"Completed in {elapsed:.2f}s")
    print(f"Total conversations: {report['total_conversations']}")
    print(f"Total messages segmented: {report['total_messages_segmented']}")
    print(f"Total turns: {report['total_turns']}")
    print("Averages:", report["averages"])
    print("Size distribution:", report["conversation_size_distribution"])
    print("Duration distribution:", report["conversation_duration_distribution"])
    print("Top initiators:", report["top_initiators"])

if __name__ == "__main__":
    main()
