import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import clean_dataset

def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    input_file = "data/processed/sanitized_messages.jsonl"
    output_file = "data/processed/cleaned_messages.jsonl"
    report_file = "data/processed/cleaning_report.json"

    print(f"Starting T006 Text Cleaning over {input_file}...")
    start_t = time.time()
    report = clean_dataset(input_file, output_file, report_file)
    elapsed = time.time() - start_t

    print(f"Completed in {elapsed:.2f}s")
    print(f"Total processed: {report['total_messages_processed']}")
    print(f"Total tokens: {report['total_tokens_extracted']}")
    print(f"Avg tokens/message: {report['avg_tokens_per_message']}")
    print("Style distribution:", report["style_distributions"])
    print("Top emojis:", list(report["top_15_emojis"].items())[:5])
    print("Top elongated words:", list(report["top_15_elongated_words"].items())[:5])
    print("Top all-caps words:", list(report["top_15_all_caps_words"].items())[:5])

if __name__ == "__main__":
    main()
