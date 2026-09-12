"""
Robust WhatsApp TXT export parser.

T002 responsibilities:
- Detect message boundaries from valid WhatsApp headers.
- Preserve message text and physical line provenance.
- Classify common WhatsApp export message types.
- Treat [Forwarded] as a per-message attribute, never as a block marker.
- Support parsing one file or every .txt file in a raw-data directory.

No cleaning, privacy sanitization, persona inference, or semantic analysis belongs here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import json
import re
from typing import Iterable, Iterator, Optional


# Covers common WhatsApp exports such as:
# [9/5/26, 7:31:22 PM] You: hello
# [05/09/2026, 19:31:22] You: hello
# [9/5/26, 7:31 PM] You: hello
HEADER_RE = re.compile(
    r"^\[(?P<date>\d{1,4}[/-]\d{1,2}[/-]\d{1,4}),\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)\]\s*"
    r"(?P<body>.*)$"
)

# WhatsApp's common export media/system markers.
MEDIA_PATTERNS = (
    ("image", re.compile(r"^<image omitted>$", re.I)),
    ("video", re.compile(r"^<video omitted>$", re.I)),
    ("gif", re.compile(r"^<gif omitted>$", re.I)),
    ("audio", re.compile(r"^<(?:audio|voice message) omitted>$", re.I)),
    ("sticker", re.compile(r"^<sticker omitted>$", re.I)),
    ("album", re.compile(r"^<album message>$", re.I)),
    ("document", re.compile(r"^<document omitted>(?:\s+.*)?$", re.I)),
)

DELETED_RE = re.compile(r"^This message was deleted$", re.I)
CALL_RE = re.compile(r"^-\s*\[Call\]\s*$", re.I)
SYSTEM_PREFIXES = (
    "Messages and calls are end-to-end encrypted",
    "You created this group",
    "You added",
    "added you",
    "changed the subject",
    "changed this group's icon",
    "changed the group description",
    "left",
    "removed",
)

# Export headers can have dates in different orders. We deliberately parse
# conservatively rather than guessing silently.
DATE_FORMATS = (
    "%m/%d/%y",
    "%m/%d/%Y",
    "%d/%m/%y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%Y-%m-%d",
    "%m-%d-%y",
    "%m-%d-%Y",
    "%d-%m-%y",
    "%d-%m-%Y",
)
TIME_FORMATS = ("%I:%M:%S %p", "%I:%M %p", "%H:%M:%S", "%H:%M")


@dataclass(frozen=True)
class ParsedMessage:
    message_id: int
    timestamp: Optional[str]
    sender: Optional[str]
    text: str
    message_type: str
    is_system: bool
    is_media: bool
    is_forwarded: bool
    raw_line_start: int
    raw_line_end: int
    source_file: str
    parse_warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        value = asdict(self)
        value["parse_warnings"] = list(self.parse_warnings)
        return value


@dataclass(frozen=True)
class ParseIssue:
    source_file: str
    line_number: int
    line: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParseResult:
    messages: list[ParsedMessage]
    issues: list[ParseIssue]
    source_file: str

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "message_count": len(self.messages),
            "issue_count": len(self.issues),
            "messages": [m.to_dict() for m in self.messages],
            "issues": [i.to_dict() for i in self.issues],
        }


def _parse_timestamp(date_text: str, time_text: str) -> tuple[Optional[str], Optional[str]]:
    """Return ISO timestamp and warning, if any."""
    normalized_time = re.sub(r"\s+", " ", time_text.strip()).upper()

    date_obj = None
    for fmt in DATE_FORMATS:
        try:
            date_obj = datetime.strptime(date_text, fmt).date()
            break
        except ValueError:
            continue

    if date_obj is None:
        return None, f"unrecognized date format: {date_text!r}"

    time_obj = None
    for fmt in TIME_FORMATS:
        try:
            time_obj = datetime.strptime(normalized_time, fmt).time()
            break
        except ValueError:
            continue

    if time_obj is None:
        return None, f"unrecognized time format: {time_text!r}"

    return datetime.combine(date_obj, time_obj).isoformat(), None


def _classify(text: str) -> tuple[str, bool, bool]:
    stripped = text.strip()
    # [Forwarded] is metadata attached to this message. Remove only that
    # prefix for media classification; never remove it from the stored text.
    classification_text = re.sub(r"^\[Forwarded\]\s*", "", stripped, count=1, flags=re.I)

    if CALL_RE.fullmatch(classification_text):
        return "call", True, False

    if DELETED_RE.fullmatch(classification_text):
        return "deleted", False, False

    for message_type, pattern in MEDIA_PATTERNS:
        if pattern.fullmatch(classification_text):
            return message_type, False, True

    # WhatsApp system messages have no sender in the usual export format.
    if any(classification_text.startswith(prefix) for prefix in SYSTEM_PREFIXES):
        return "system", True, False

    return "text", False, False


def _split_sender(body: str) -> tuple[Optional[str], str]:
    """
    Split the header body into sender/content.

    We use the first ': ' / ':' delimiter rather than a greedy expression,
    because message text itself may contain colons.
    """
    if ": " in body:
        sender, text = body.split(": ", 1)
        return sender, text

    # Some exports omit the space after ':'.
    if ":" in body:
        sender, text = body.split(":", 1)
        if sender and text:
            return sender, text.lstrip()

    return None, body


def _iter_lines(text: str) -> Iterator[str]:
    # splitlines() handles LF, CRLF and CR without altering message contents.
    yield from text.splitlines()


class WhatsAppParser:
    """Parser for one or many WhatsApp TXT exports."""

    def __init__(self, *, strict: bool = False, encoding: str = "utf-8") -> None:
        self.strict = strict
        self.encoding = encoding

    def parse_text(self, text: str, source_file: str = "<memory>") -> ParseResult:
        messages: list[ParsedMessage] = []
        issues: list[ParseIssue] = []

        current_header: Optional[dict] = None
        current_text_lines: list[str] = []

        def flush() -> None:
            nonlocal current_header, current_text_lines

            if current_header is None:
                return

            raw_text = "\n".join(current_text_lines)
            timestamp, timestamp_warning = _parse_timestamp(
                current_header["date"], current_header["time"]
            )

            warnings: list[str] = []
            if timestamp_warning:
                warnings.append(timestamp_warning)

            message_type, is_system, is_media = _classify(raw_text)

            sender = current_header["sender"]
            # A normal system/call line may have no sender.
            if sender is None and message_type not in {"system", "call"}:
                warnings.append("message header has no sender delimiter")

            messages.append(
                ParsedMessage(
                    message_id=len(messages) + 1,
                    timestamp=timestamp,
                    sender=sender,
                    text=raw_text,
                    message_type=message_type,
                    is_system=is_system,
                    is_media=is_media,
                    is_forwarded=raw_text.startswith("[Forwarded]"),
                    raw_line_start=current_header["line_number"],
                    raw_line_end=current_header["last_line_number"],
                    source_file=source_file,
                    parse_warnings=tuple(warnings),
                )
            )

            current_header = None
            current_text_lines = []

        for line_number, line in enumerate(_iter_lines(text), start=1):
            match = HEADER_RE.match(line)

            if match:
                flush()

                sender, body = _split_sender(match.group("body"))
                current_header = {
                    "date": match.group("date"),
                    "time": match.group("time"),
                    "sender": sender,
                    "line_number": line_number,
                    "last_line_number": line_number,
                }
                current_text_lines = [body]
                continue

            # Anything after a valid header belongs to the current message.
            if current_header is not None:
                current_text_lines.append(line)
                current_header["last_line_number"] = line_number
                continue

            # Ignore only completely blank preamble lines.
            if line.strip():
                issue = ParseIssue(
                    source_file=source_file,
                    line_number=line_number,
                    line=line,
                    reason="orphan line before first valid WhatsApp message header",
                )
                issues.append(issue)
                if self.strict:
                    raise ValueError(
                        f"{source_file}:{line_number}: {issue.reason}: {line!r}"
                    )

        flush()

        return ParseResult(messages=messages, issues=issues, source_file=source_file)

    def parse_file(self, path: str | Path) -> ParseResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"WhatsApp export not found: {path}")
        if not path.is_file():
            raise ValueError(f"Expected a file, got: {path}")

        try:
            text = path.read_text(encoding=self.encoding)
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Could not decode {path} as {self.encoding}; "
                "try a different encoding."
            ) from exc

        return self.parse_text(text, source_file=path.name)

    def parse_directory(
        self,
        raw_dir: str | Path,
        *,
        recursive: bool = False,
        pattern: str = "*.txt",
    ) -> list[ParseResult]:
        """
        Parse every TXT export in raw_dir.

        This is the entry point intended for the project:
        one export == one conversation source, but all exports can later be
        combined by the pipeline without losing source_file provenance.
        """
        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
        if not raw_dir.is_dir():
            raise ValueError(f"Expected a directory, got: {raw_dir}")

        paths = sorted(raw_dir.rglob(pattern) if recursive else raw_dir.glob(pattern))
        return [self.parse_file(path) for path in paths]


def parse_whatsapp_file(path: str | Path) -> ParseResult:
    return WhatsAppParser().parse_file(path)


def parse_raw_directory(
    raw_dir: str | Path,
    *,
    recursive: bool = False,
) -> list[ParseResult]:
    return WhatsAppParser().parse_directory(raw_dir, recursive=recursive)


def save_results_json(results: Iterable[ParseResult], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.to_dict() for result in results]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    import argparse

    cli = argparse.ArgumentParser(description="Parse WhatsApp TXT exports.")
    cli.add_argument("raw_dir", help="Directory containing WhatsApp .txt exports")
    cli.add_argument(
        "--recursive",
        action="store_true",
        help="Search nested directories for .txt exports",
    )
    cli.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path",
    )
    args = cli.parse_args()

    parser = WhatsAppParser()
    results = parser.parse_directory(args.raw_dir, recursive=args.recursive)

    summary = {
        result.source_file: {
            "messages": len(result.messages),
            "issues": len(result.issues),
        }
        for result in results
    }
    print(json.dumps(summary, indent=2))

    if args.output:
        save_results_json(results, args.output)
