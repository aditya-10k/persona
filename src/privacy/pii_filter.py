"""
Stage T005: Privacy Filter

Detects and masks PII (Personally Identifiable Information) and sensitive data
from message text while strictly preserving:
- sender identity (needed for downstream speaker tracking and cross-chat analysis)
- stylistic markers (emojis, slang, Hinglish, punctuation, capitalization, repeated letters)
- full canonical schema and line provenance
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class PrivacyFilter:
    """
    Detects and sanitizes sensitive PII in chat messages.
    """

    # 1. URLs and Web links
    URL_PATTERN = re.compile(
        r'(?:https?://|ftp://|www\.)[^\s<>"\'{}|\\^`]+|'
        r'(?:(?:[a-zA-Z0-9-]+\.)+(?:com|org|net|edu|gov|io|ai|me|app|co|in|dev|ly|be|gl|is))/[^\s<>"\'{}|\\^`]+',
        re.IGNORECASE,
    )

    # 2. Email addresses
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    )

    # 3. UPI IDs and payment URIs
    UPI_HANDLE_PATTERN = re.compile(
        r'\b[a-zA-Z0-9.\-_]{2,}@(okaxis|okhdfcbank|okicici|oksbi|paytm|ybl|axl|ibl|apl|upi|postbank|federal|icici|hdfcbank|sbi|kotak)\b',
        re.IGNORECASE,
    )
    UPI_URI_PATTERN = re.compile(r'upi://pay\?[^\s<>"\'{}|\\^`]+', re.IGNORECASE)

    # 4. API keys and well-known credential tokens
    API_KEY_PATTERNS = [
        re.compile(r'\bsk-[a-zA-Z0-9]{20,}\b'),  # OpenAI
        re.compile(r'\bAIza[0-9A-Za-z-_]{30,45}\b'),  # Google API
        re.compile(r'\b(?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36}\b'),  # GitHub
        re.compile(r'\bxox[baprs]-[0-9a-zA-Z]{10,48}\b'),  # Slack
        re.compile(
            r'\beyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b'
        ),  # JWT
    ]

    # 5. Explicit password / credential declarations
    PASSWORD_CONTEXT_PATTERN = re.compile(
        r'(?i)\b(?:password|passwd|pwd|passcode)\s*(?:is|:|=|-)\s*([^\s\n,.]+)'
    )

    # Credential line directly following an email (common WhatsApp forwarded credential pattern)
    EMAIL_FOLLOWED_BY_SECRET = re.compile(
        r'(\[EMAIL\]\s*\n\s*)([A-Za-z0-9!@#$%^&*()_+\-=]{6,}\b)'
    )

    # 6. Financial card numbers (16 digits with optional spaces or hyphens)
    CARD_PATTERN = re.compile(r'\b(?:\d{4}[ -]?){3}\d{4}\b')

    # 7. OTPs and verification codes
    OTP_CONTEXT_PATTERN = re.compile(
        r'(?i)\b(?:otp|one[- ]time[- ]password|verification\s*code|auth\s*code|security\s*code|passcode)\s*(?:is|:|=|-)?\s*([0-9]{4,8})\b'
    )
    OTP_ACTION_PATTERN = re.compile(
        r'(?i)\b(?:use|enter|your)\s+(?:code|otp|pin)\s*[:=\-]?\s*(?:is\s+)?([0-9]{4,8})\b'
    )

    # 8. Phone numbers (Indian mobile and international formats)
    # Target 10-15 digits, optionally with country code (+91, etc.)
    # Indian mobile: starting with 6, 7, 8, or 9
    PHONE_INDIAN_PATTERN = re.compile(
        r'(?:\+91[\s\-]?)?(?:\b0)?[6-9]\d{4}[\s\-]?[0-9]{5}\b|'
        r'(?:\+91[\s\-]?)?(?:\b0)?[6-9]\d{2}[\s\-]?[0-9]{3}[\s\-]?[0-9]{4}\b|'
        r'(?:\+91[\s\-]?)?(?:\b0)?[6-9]\d{9}\b'
    )
    # General international number with explicit leading + and country code
    PHONE_INTL_PATTERN = re.compile(
        r'\+\d{1,3}[\s\-]?(?:\(\d{1,4}\)[\s\-]?)?\d{2,4}[\s\-]?\d{3,4}[\s\-]?\d{3,4}\b'
    )

    def mask_pii(self, text: str) -> Tuple[str, List[str]]:
        """
        Sanitizes text by replacing detected PII with standardized tokens.
        Preserves slang, Hinglish, emojis, casing, and stylistic punctuation.

        Returns:
            (sanitized_text, detected_pii_categories)
        """
        if not text:
            return text, []

        detected_categories: Set[str] = set()
        sanitized = text

        # 1. Mask URLs
        def _replace_url(match: re.Match) -> str:
            val = match.group(0)
            # Preserve trailing punctuation attached to the URL
            trailing = ""
            while val and val[-1] in ".,!?:;)>]}'\"":
                trailing = val[-1] + trailing
                val = val[:-1]
            return "[URL]" + trailing

        if self.URL_PATTERN.search(sanitized):
            sanitized = self.URL_PATTERN.sub(_replace_url, sanitized)
            detected_categories.add("url")

        # 2. Mask Emails
        if self.EMAIL_PATTERN.search(sanitized):
            sanitized = self.EMAIL_PATTERN.sub("[EMAIL]", sanitized)
            detected_categories.add("email")

        # 3. Mask UPI Handles and URIs
        if self.UPI_URI_PATTERN.search(sanitized):
            sanitized = self.UPI_URI_PATTERN.sub("[UPI_ID]", sanitized)
            detected_categories.add("upi_id")

        if self.UPI_HANDLE_PATTERN.search(sanitized):
            sanitized = self.UPI_HANDLE_PATTERN.sub("[UPI_ID]", sanitized)
            detected_categories.add("upi_id")

        # 4. Mask API Keys & Known Tokens
        for pattern in self.API_KEY_PATTERNS:
            if pattern.search(sanitized):
                sanitized = pattern.sub("[CREDENTIAL]", sanitized)
                detected_categories.add("credential")

        # 5. Mask Explicit Passwords
        if self.PASSWORD_CONTEXT_PATTERN.search(sanitized):
            sanitized = self.PASSWORD_CONTEXT_PATTERN.sub(
                lambda m: m.group(0).replace(m.group(1), "[CREDENTIAL]"),
                sanitized,
            )
            detected_categories.add("credential")

        if self.EMAIL_FOLLOWED_BY_SECRET.search(sanitized):
            sanitized = self.EMAIL_FOLLOWED_BY_SECRET.sub(
                r"\1[CREDENTIAL]", sanitized
            )
            detected_categories.add("credential")

        # 6. Mask Financial Card Numbers (16 digits)
        def _replace_card(match: re.Match) -> str:
            digits_only = re.sub(r"\D", "", match.group(0))
            if len(digits_only) == 16:
                detected_categories.add("financial_id")
                return "[FINANCIAL_ID]"
            return match.group(0)

        sanitized = self.CARD_PATTERN.sub(_replace_card, sanitized)

        # 7. Mask OTPs & Verification Codes
        if self.OTP_CONTEXT_PATTERN.search(sanitized):
            sanitized = self.OTP_CONTEXT_PATTERN.sub(
                lambda m: m.group(0).replace(m.group(1), "[OTP]"), sanitized
            )
            detected_categories.add("otp")

        if self.OTP_ACTION_PATTERN.search(sanitized):
            sanitized = self.OTP_ACTION_PATTERN.sub(
                lambda m: m.group(0).replace(m.group(1), "[OTP]"), sanitized
            )
            detected_categories.add("otp")

        # 8. Mask Phone Numbers
        # International with +
        def _replace_intl_phone(match: re.Match) -> str:
            raw = match.group(0)
            digits = re.sub(r"\D", "", raw)
            if 10 <= len(digits) <= 15:
                detected_categories.add("phone_number")
                return "[PHONE_NUMBER]"
            return raw

        sanitized = self.PHONE_INTL_PATTERN.sub(_replace_intl_phone, sanitized)

        # Indian phone numbers
        def _replace_indian_phone(match: re.Match) -> str:
            raw = match.group(0)
            digits = re.sub(r"\D", "", raw)
            # Indian numbers: 10 digits (or 11 with leading 0, or 12 with 91)
            if (
                len(digits) == 10
                or (len(digits) == 11 and digits.startswith("0"))
                or (len(digits) == 12 and digits.startswith("91"))
            ):
                detected_categories.add("phone_number")
                return "[PHONE_NUMBER]"
            return raw

        sanitized = self.PHONE_INDIAN_PATTERN.sub(_replace_indian_phone, sanitized)

        return sanitized, sorted(list(detected_categories))

    def clean_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitizes a single canonical message record while strictly preserving
        the sender field, message metadata, and source provenance.
        """
        original_text = record.get("text", "")
        sanitized_text, detected = self.mask_pii(original_text)

        # Build clean record preserving all original fields
        sanitized_record = dict(record)
        # Explicitly preserve sender exactly as original
        sanitized_record["sender"] = record.get("sender")
        sanitized_record["text"] = sanitized_text
        sanitized_record["original_text_masked"] = (sanitized_text != original_text)
        sanitized_record["pii_detected"] = detected

        return sanitized_record


def sanitize_dataset(
    input_jsonl_path: str,
    output_jsonl_path: str,
    report_output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Streams and sanitizes the canonical message dataset.
    Generates sanitized_messages.jsonl and privacy_report.json.
    """
    in_path = Path(input_jsonl_path)
    out_path = Path(output_jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    privacy_filter = PrivacyFilter()

    total_records = 0
    sanitized_records = 0
    pii_counts: Dict[str, int] = {}
    records_by_source: Dict[str, Dict[str, int]] = {}

    with open(in_path, "r", encoding="utf-8") as in_f, open(
        out_path, "w", encoding="utf-8"
    ) as out_f:
        for line in in_f:
            line_str = line.strip()
            if not line_str:
                continue

            record = json.loads(line_str)
            total_records += 1

            clean_rec = privacy_filter.clean_record(record)
            out_f.write(json.dumps(clean_rec, ensure_ascii=False) + "\n")

            if clean_rec["original_text_masked"]:
                sanitized_records += 1

            source = clean_rec.get("source_file", "unknown")
            if source not in records_by_source:
                records_by_source[source] = {"total": 0, "sanitized": 0}
            records_by_source[source]["total"] += 1
            if clean_rec["original_text_masked"]:
                records_by_source[source]["sanitized"] += 1

            for pii_type in clean_rec["pii_detected"]:
                pii_counts[pii_type] = pii_counts.get(pii_type, 0) + 1

    report = {
        "schema_version": "1.0",
        "task_id": "T005",
        "status": "COMPLETE",
        "total_messages_processed": total_records,
        "total_messages_sanitized": sanitized_records,
        "total_messages_unchanged": total_records - sanitized_records,
        "pii_detections_by_type": pii_counts,
        "records_by_source_file": records_by_source,
    }

    if report_output_path:
        rep_path = Path(report_output_path)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rep_path, "w", encoding="utf-8") as rep_f:
            json.dump(report, rep_f, indent=2, ensure_ascii=False)

    return report
