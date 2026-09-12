import json
from pathlib import Path

def main():
    json_path = Path("data/processed/pii_detections_non_url.json")
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    md_lines = [
        "# Non-URL PII Detections Audit Document",
        "",
        "This document contains all **135 messages** where non-URL sensitive data was detected and masked by Stage T005.",
        "",
        "### Summary Breakdown:",
        "- **Credentials / Secrets / API Keys**: 64",
        "- **Email Addresses**: 63",
        "- **Phone Numbers**: 59",
        "- **Financial Card Identifiers**: 3",
        "- **UPI Payment IDs**: 2",
        "- **OTPs / Verification Codes**: 1",
        "",
        "---",
        "",
    ]

    for idx, item in enumerate(items, 1):
        pii_types = ", ".join(item["non_url_pii"])
        md_lines.append(f"### {idx}. [{item['message_id']}] ({item['source_file']})")
        md_lines.append(f"- **Sender**: `{item['sender']}`")
        md_lines.append(f"- **Timestamp**: `{item['timestamp']}`")
        md_lines.append(f"- **PII Detected**: `{pii_types}`")
        md_lines.append("")
        md_lines.append("**Original Text**:")
        md_lines.append("```text")
        md_lines.append(item["orig_text"])
        md_lines.append("```")
        md_lines.append("")
        md_lines.append("**Sanitized Text**:")
        md_lines.append("```text")
        md_lines.append(item["sanitized_text"])
        md_lines.append("```")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

    out_path = Path("data/processed/non_url_pii_audit.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Generated {out_path} with {len(items)} entries.")

if __name__ == "__main__":
    main()
