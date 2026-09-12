"""
Unit and integration tests for T005 Privacy Filter.
"""

import json
from pathlib import Path
import pytest

from src.privacy.pii_filter import PrivacyFilter, sanitize_dataset


@pytest.fixture
def pii_filter():
    return PrivacyFilter()


def test_email_masking(pii_filter):
    text = "Please reach out to contact.work@example.com or john.doe@work.co.uk for details."
    sanitized, detected = pii_filter.mask_pii(text)
    assert "[EMAIL]" in sanitized
    assert "contact.work@example.com" not in sanitized
    assert "john.doe@work.co.uk" not in sanitized
    assert "email" in detected


def test_url_masking(pii_filter):
    text = "Watch this reel: https://www.instagram.com/reel/DVPRP6Zkuxf/?igsh=abc and check https://meet.google.com/xyz-abc-def."
    sanitized, detected = pii_filter.mask_pii(text)
    assert "[URL]" in sanitized
    assert "instagram.com" not in sanitized
    assert "meet.google.com" not in sanitized
    assert sanitized.endswith(".")  # Trailing period preserved
    assert "url" in detected


def test_phone_number_masking(pii_filter):
    # Indian mobile with +91 and spaces
    text1 = "Call me on +91 98765 43210 later."
    sanitized1, detected1 = pii_filter.mask_pii(text1)
    assert sanitized1 == "Call me on [PHONE_NUMBER] later."
    assert "phone_number" in detected1

    # Indian 10 digits continuous
    text2 = "Number is 9876543210 please save."
    sanitized2, detected2 = pii_filter.mask_pii(text2)
    assert sanitized2 == "Number is [PHONE_NUMBER] please save."
    assert "phone_number" in detected2

    # Spaced 5-5
    text3 = "Here: 91234 56789"
    sanitized3, detected3 = pii_filter.mask_pii(text3)
    assert sanitized3 == "Here: [PHONE_NUMBER]"
    assert "phone_number" in detected3


def test_upi_id_masking(pii_filter):
    text = "Send 500 to rahul@okaxis or payments@paytm."
    sanitized, detected = pii_filter.mask_pii(text)
    assert "rahul@okaxis" not in sanitized
    assert "payments@paytm" not in sanitized
    assert "[UPI_ID]" in sanitized
    assert "upi_id" in detected


def test_credentials_and_passwords(pii_filter):
    # Explicit password context
    text1 = "password is SecretPasswd987"
    sanitized1, detected1 = pii_filter.mask_pii(text1)
    assert "SecretPasswd987" not in sanitized1
    assert "[CREDENTIAL]" in sanitized1
    assert "credential" in detected1

    # API key
    text2 = "Here is the key: AIzaSyD3fakeAPIkey1234567890abcdefgh"
    sanitized2, detected2 = pii_filter.mask_pii(text2)
    assert "AIzaSyD" not in sanitized2
    assert "[CREDENTIAL]" in sanitized2
    assert "credential" in detected2

    # Email followed by password on newline (common forwarded credential format)
    text3 = "[Forwarded] contact.work@example.com\nSecretPasswd987"
    sanitized3, detected3 = pii_filter.mask_pii(text3)
    assert "contact.work@example.com" not in sanitized3
    assert "SecretPasswd987" not in sanitized3
    assert "[EMAIL]" in sanitized3
    assert "[CREDENTIAL]" in sanitized3
    assert "email" in detected3
    assert "credential" in detected3


def test_otp_masking(pii_filter):
    text1 = "Your OTP is 491023 for logging in."
    sanitized1, detected1 = pii_filter.mask_pii(text1)
    assert "491023" not in sanitized1
    assert "[OTP]" in sanitized1
    assert "otp" in detected1

    text2 = "Enter code: 8829"
    sanitized2, detected2 = pii_filter.mask_pii(text2)
    assert "8829" not in sanitized2
    assert "[OTP]" in sanitized2
    assert "otp" in detected2


def test_financial_card_masking(pii_filter):
    text = "Card is 4111 2222 3333 4444 exp 12/28"
    sanitized, detected = pii_filter.mask_pii(text)
    assert "4111 2222 3333 4444" not in sanitized
    assert "[FINANCIAL_ID]" in sanitized
    assert "financial_id" in detected


def test_preserves_slang_emojis_hinglish_and_style(pii_filter):
    """
    CRITICAL: Verify stylistic markers are NOT falsely masked.
    """
    sample_texts = [
        "broooo this is FUCKED 😭😭",
        "kya haal hai bhaiii, kal milte hai",
        "lol ngl tbh idk why this happened",
        "Password toh mai bhi bhool gaya hu 💀",
        "Are chat me hoga na bhai",
        "WHAT?! Are you serious?? ...",
        "Meeting at 10:30, wait for 5 mins, 2026 release",
    ]

    for text in sample_texts:
        sanitized, detected = pii_filter.mask_pii(text)
        # Should not mask words as PII
        assert detected == [], f"False positive PII detected in: {text} -> {detected}"
        assert sanitized == text, f"Text altered unexpectedly: {sanitized}"


def test_clean_record_preserves_sender_and_provenance(pii_filter):
    record = {
        "message_id": "chat_01.txt:00000008",
        "timestamp": "2026-05-16T18:39:12",
        "sender": "+91 91234 56789",  # Sender must remain strictly intact
        "text": "Call me on 9876543210 please",
        "message_type": "text",
        "is_system": False,
        "is_media": False,
        "is_forwarded": False,
        "source_file": "chat_01.txt",
        "raw_line_start": 9,
        "raw_line_end": 10,
        "parse_warnings": [],
    }

    clean = pii_filter.clean_record(record)

    # Sender strictly unchanged
    assert clean["sender"] == "+91 91234 56789"
    # Text sanitized
    assert clean["text"] == "Call me on [PHONE_NUMBER] please"
    assert clean["original_text_masked"] is True
    assert clean["pii_detected"] == ["phone_number"]
    # Provenance unchanged
    assert clean["message_id"] == "chat_01.txt:00000008"
    assert clean["source_file"] == "chat_01.txt"
    assert clean["raw_line_start"] == 9


def test_sanitize_dataset(tmp_path):
    input_file = tmp_path / "messages.jsonl"
    output_file = tmp_path / "sanitized_messages.jsonl"
    report_file = tmp_path / "privacy_report.json"

    records = [
        {
            "message_id": "test:001",
            "timestamp": "2026-05-16T18:38:11",
            "sender": "You",
            "text": "broooo email is test@example.com 😭😭",
            "message_type": "text",
            "is_system": False,
            "is_media": False,
            "is_forwarded": False,
            "source_file": "test.txt",
            "raw_line_start": 1,
            "raw_line_end": 1,
            "parse_warnings": [],
        },
        {
            "message_id": "test:002",
            "timestamp": "2026-05-16T18:38:17",
            "sender": "Friend_01",
            "text": "bhaiii all good no pii here",
            "message_type": "text",
            "is_system": False,
            "is_media": False,
            "is_forwarded": False,
            "source_file": "test.txt",
            "raw_line_start": 2,
            "raw_line_end": 2,
            "parse_warnings": [],
        },
    ]

    with open(input_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    report = sanitize_dataset(str(input_file), str(output_file), str(report_file))

    assert report["total_messages_processed"] == 2
    assert report["total_messages_sanitized"] == 1
    assert report["total_messages_unchanged"] == 1
    assert report["pii_detections_by_type"] == {"email": 1}

    # Verify output file
    with open(output_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 2
    assert lines[0]["text"] == "broooo email is [EMAIL] 😭😭"
    assert lines[0]["sender"] == "You"
    assert lines[1]["text"] == "bhaiii all good no pii here"
    assert lines[1]["sender"] == "Friend_01"
