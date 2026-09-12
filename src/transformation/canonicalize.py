from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MESSAGE_TYPE_MAP = {
    "text": "text",
    "image": "image",
    "video": "video",
    "gif": "gif",
    "audio": "audio",
    "voice": "audio",
    "sticker": "sticker",
    "document": "document",
    "album": "album",
    "deleted": "deleted",
    "call": "call",
    "system": "system",
}


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode without destroying emojis or intentional characters.

    NFC is deliberately used instead of aggressive compatibility
    normalization because style-sensitive text must remain recoverable.
    """
    return unicodedata.normalize("NFC", text)


def normalize_message_type(message_type: str | None) -> str:
    if not message_type:
        return "text"

    normalized = message_type.strip().lower()
    return MESSAGE_TYPE_MAP.get(normalized, normalized)


def canonicalize_message(
    message: dict[str, Any],
    source_file: str,
    local_index: int,
) -> dict[str, Any]:
    """
    Convert one parser record into the canonical T003 representation.
    """

    raw_text = message.get("text", "")

    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    # Important:
    # Do NOT strip/collapse whitespace here.
    # The original text is a style signal.
    text = normalize_unicode(raw_text)

    message_type = normalize_message_type(
        message.get("message_type")
    )

    # Stable across repeated runs as long as the source file and
    # parser ordering remain unchanged.
    message_id = (
        f"{source_file}:{local_index:08d}"
    )

    return {
        "message_id": message_id,
        "timestamp": message.get("timestamp"),
        "sender": message.get("sender"),
        "text": text,
        "message_type": message_type,
        "is_system": bool(message.get("is_system", False)),
        "is_media": bool(message.get("is_media", False)),
        "is_forwarded": bool(message.get("is_forwarded", False)),
        "source_file": source_file,
        "raw_line_start": message.get("raw_line_start"),
        "raw_line_end": message.get("raw_line_end"),
        "parse_warnings": message.get("parse_warnings", []),
    }


def load_parsed_messages(
    parsed_path: Path,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """
    Load the exact output contract produced by T002.

    T002 output:

    [
        {
            "source_file": "chat_01.txt",
            "message_count": 1676,
            "issue_count": 1,
            "messages": [...]
        },
        ...
    ]

    Returns:
        [
            ("chat_01.txt", [...]),
            ("chat_02.txt", [...]),
            ...
        ]
    """

    with parsed_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Parsed dataset must be a list of per-file results."
        )

    result = []

    for file_result in data:

        if not isinstance(file_result, dict):
            raise ValueError(
                "Each parsed file result must be an object."
            )

        source_file = file_result.get("source_file")

        if not source_file:
            raise ValueError(
                "Parsed file result is missing 'source_file'."
            )

        messages = file_result.get("messages", [])

        if not isinstance(messages, list):
            raise ValueError(
                f"'messages' must be a list for {source_file}"
            )

        result.append(
            (
                str(source_file),
                messages,
            )
        )

    # T002 already produces deterministic file ordering,
    # but sorting here makes T003's behavior independent
    # of input ordering.
    result.sort(
        key=lambda item: item[0]
    )

    return result


def build_sender_stats(
    canonical_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Generate aggregate sender statistics.

    Statistics are descriptive only.
    They do not infer personality or behavior.
    """

    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "message_count": 0,
            "text_message_count": 0,
            "media_message_count": 0,
            "system_message_count": 0,
            "forwarded_message_count": 0,
            "message_types": {},
            "source_files": {},
            "total_characters": 0,
            "total_words": 0,
        }
    )

    for message in canonical_messages:
        sender = message.get("sender")

        if sender is None:
            sender = "<UNKNOWN>"

        sender = str(sender)

        entry = stats[sender]

        entry["message_count"] += 1

        if message["message_type"] == "text":
            entry["text_message_count"] += 1

        if message["is_media"]:
            entry["media_message_count"] += 1

        if message["is_system"]:
            entry["system_message_count"] += 1

        if message["is_forwarded"]:
            entry["forwarded_message_count"] += 1

        message_type = message["message_type"]

        type_counter = Counter(
            entry["message_types"]
        )

        type_counter[message_type] += 1

        entry["message_types"] = dict(
            sorted(type_counter.items())
        )

        text = message["text"]

        entry["total_characters"] += len(text)

        entry["total_words"] += len(
            text.split()
        )

        source_file = message["source_file"]

        source_counter = Counter(
            entry["source_files"]
        )

        source_counter[source_file] += 1

        entry["source_files"] = dict(
            sorted(source_counter.items())
        )

    # Derived averages.
    for sender in stats:
        entry = stats[sender]

        count = entry["message_count"]

        entry["average_characters_per_message"] = (
            round(
                entry["total_characters"] / count,
                2,
            )
            if count
            else 0
        )

        entry["average_words_per_message"] = (
            round(
                entry["total_words"] / count,
                2,
            )
            if count
            else 0
        )

    return {
        "total_unique_senders": len(stats),
        "senders": dict(
            sorted(stats.items())
        ),
    }


def transform(
    parsed_path: Path,
    output_jsonl: Path,
    sender_stats_path: Path,
) -> dict[str, Any]:

    parsed_files = load_parsed_messages(
        parsed_path
    )

    canonical_messages: list[dict[str, Any]] = []

    per_file_counts = {}

    for source_file, messages in parsed_files:

        per_file_counts[source_file] = len(messages)

        for index, message in enumerate(
            messages,
            start=1,
        ):
            canonical = canonicalize_message(
                message=message,
                source_file=source_file,
                local_index=index,
            )

            canonical_messages.append(
                canonical
            )

    output_jsonl.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sender_stats_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # JSONL: one canonical message per line.
    with output_jsonl.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:

        for message in canonical_messages:
            f.write(
                json.dumps(
                    message,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    sender_stats = build_sender_stats(
        canonical_messages
    )

    with sender_stats_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:

        json.dump(
            sender_stats,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "files_processed": len(parsed_files),
        "total_messages": len(
            canonical_messages
        ),
        "unique_senders": sender_stats[
            "total_unique_senders"
        ],
        "messages_by_file": per_file_counts,
        "output": str(output_jsonl),
        "sender_stats": str(
            sender_stats_path
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transform parsed WhatsApp messages "
            "into the canonical T003 dataset."
        )
    )

    parser.add_argument(
        "parsed",
        type=Path,
        help="Path to parsed.json",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/messages.jsonl"
        ),
    )

    parser.add_argument(
        "--sender-stats",
        type=Path,
        default=Path(
            "data/processed/sender_stats.json"
        ),
    )

    args = parser.parse_args()

    result = transform(
        parsed_path=args.parsed,
        output_jsonl=args.output,
        sender_stats_path=args.sender_stats,
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()