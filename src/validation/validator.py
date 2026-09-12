from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


# These are the message types produced/allowed by T002/T003.
KNOWN_MESSAGE_TYPES = {
    "text",
    "image",
    "video",
    "gif",
    "audio",
    "sticker",
    "document",
    "album",
    "deleted",
    "call",
    "system",
}


# A message above this size is suspicious, but NOT automatically invalid.
LARGE_MESSAGE_THRESHOLD = 20_000


# WhatsApp adds this message to exported chats.
# It has no sender and therefore should not be treated as a
# genuine missing-sender conversation message.
ENCRYPTION_PREAMBLE = (
    "- Messages and calls are end-to-end encrypted. "
    "No one outside of this chat, not even WhatsApp, can read "
    "or listen to them. Tap to learn more."
)


# --------------------------------------------------------------------
# System-generated WhatsApp events
# --------------------------------------------------------------------
#
# These messages legitimately have no sender in the TXT export.
#
# Examples observed in the dataset:
#
#   - [Call]
#   - [System notification]
#   - User A left
#   - User B created the group
#   - You added User A
#   - Someone added User C
#   - You changed the group name to '...'
#
# These are not conversation messages authored by a sender.
# --------------------------------------------------------------------

CALL_EVENT_PREFIXES = (
    "[Call]",
    "- [Call]",
)


SYSTEM_NOTIFICATION_PREFIXES = (
    "[System notification]",
    "- [System notification]",
)


GROUP_EVENT_KEYWORDS = (
    " created the group",
    " added ",
    " removed ",
    " left",
    " changed the group name to ",
    " changed the group icon",
    " changed the group description",
    " made ",
    " made you ",
    " made You ",
    " made ",
)


# --------------------------------------------------------------------
# Parser warning that is expected for system events
# --------------------------------------------------------------------
#
# T002 currently reports:
#
#   "message header has no sender delimiter"
#
# for WhatsApp system-generated records whose header looks like:
#
#   [date, time] - [System notification]
#
# rather than:
#
#   [date, time] Sender: message
#
# This warning is expected and should NOT be treated as a parser
# failure when the corresponding record is clearly a system event.
# --------------------------------------------------------------------

EXPECTED_SYSTEM_EVENT_WARNING = (
    "message header has no sender delimiter"
)


class DatasetValidator:
    """
    T004 - Validate the canonical messages.jsonl dataset.

    This validator DOES NOT modify the dataset.

    It checks:
        - malformed JSON records
        - required fields
        - parser warnings
        - expected system-event parser warnings
        - timestamps
        - timestamp ordering
        - missing senders
        - empty messages
        - message types
        - duplicate records
        - suspiciously large messages
        - source/line provenance
        - dataset statistics
    """

    REQUIRED_FIELDS = {
        "message_id",
        "timestamp",
        "sender",
        "text",
        "message_type",
        "is_system",
        "is_media",
        "is_forwarded",
        "source_file",
        "raw_line_start",
        "raw_line_end",
        "parse_warnings",
    }

    BOOLEAN_FIELDS = {
        "is_system",
        "is_media",
        "is_forwarded",
    }

    def __init__(
        self,
        large_message_threshold: int = LARGE_MESSAGE_THRESHOLD,
    ) -> None:
        self.large_message_threshold = large_message_threshold

    # ================================================================
    # PUBLIC API
    # ================================================================

    def validate_file(
        self,
        messages_path: Path,
    ) -> dict[str, Any]:

        messages_path = Path(messages_path)

        if not messages_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {messages_path}"
            )

        records: list[dict[str, Any]] = []
        malformed_records: list[dict[str, Any]] = []

        # ------------------------------------------------------------
        # Read JSONL
        # ------------------------------------------------------------

        with messages_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line_number, line in enumerate(
                f,
                start=1,
            ):

                # Ignore blank JSONL lines.
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)

                except json.JSONDecodeError as exc:
                    malformed_records.append(
                        {
                            "jsonl_line": line_number,
                            "reason": "invalid_json",
                            "error": str(exc),
                        }
                    )
                    continue

                if not isinstance(record, dict):
                    malformed_records.append(
                        {
                            "jsonl_line": line_number,
                            "reason": "record_is_not_an_object",
                        }
                    )
                    continue

                # Internal metadata used only during validation.
                record["_jsonl_line"] = line_number

                records.append(record)

        # ------------------------------------------------------------
        # Validate records
        # ------------------------------------------------------------

        report = self._validate_records(
            records=records,
            malformed_records=malformed_records,
            messages_path=messages_path,
        )

        return report

    # ================================================================
    # CORE VALIDATION
    # ================================================================

    def _validate_records(
        self,
        records: list[dict[str, Any]],
        malformed_records: list[dict[str, Any]],
        messages_path: Path,
    ) -> dict[str, Any]:

        schema_errors: list[dict[str, Any]] = []

        # Genuine parser warnings/errors.
        parse_errors: list[dict[str, Any]] = []

        # Parser warnings that are expected because the record is
        # a legitimate WhatsApp system-generated event.
        expected_parse_warnings: list[dict[str, Any]] = []

        missing_timestamps: list[dict[str, Any]] = []
        missing_senders: list[dict[str, Any]] = []
        empty_messages: list[dict[str, Any]] = []
        unexpected_types: list[dict[str, Any]] = []
        suspicious_large_messages: list[dict[str, Any]] = []
        timestamp_anomalies: list[dict[str, Any]] = []
        provenance_errors: list[dict[str, Any]] = []

        exact_duplicate_groups: list[dict[str, Any]] = []
        potential_duplicate_groups: list[dict[str, Any]] = []

        # ------------------------------------------------------------
        # System event tracking
        # ------------------------------------------------------------

        system_events: list[dict[str, Any]] = []

        system_event_counts: Counter[str] = Counter()

        # ------------------------------------------------------------
        # Statistics
        # ------------------------------------------------------------

        messages_by_sender: Counter[str] = Counter()
        messages_by_type: Counter[str] = Counter()
        messages_by_source: Counter[str] = Counter()

        # ------------------------------------------------------------
        # Timestamp tracking
        #
        # IMPORTANT:
        # We check chronological order PER CHAT FILE.
        #
        # We must NOT compare timestamps between different chats.
        # ------------------------------------------------------------

        timestamps_by_source: defaultdict[
            str,
            list[tuple[str, datetime | None]],
        ] = defaultdict(list)

        # ------------------------------------------------------------
        # Duplicate tracking
        # ------------------------------------------------------------

        exact_duplicates: defaultdict[
            tuple[Any, ...],
            list[dict[str, Any]],
        ] = defaultdict(list)

        potential_duplicates: defaultdict[
            tuple[Any, ...],
            list[dict[str, Any]],
        ] = defaultdict(list)

        # ------------------------------------------------------------
        # Provenance tracking
        # ------------------------------------------------------------

        previous_record_by_source: dict[
            str,
            dict[str, Any],
        ] = {}

        # ============================================================
        # RECORD LOOP
        # ============================================================

        for record in records:

            # --------------------------------------------------------
            # Internal JSONL line
            # --------------------------------------------------------

            jsonl_line = record.get("_jsonl_line")

            # --------------------------------------------------------
            # 1. Required fields
            # --------------------------------------------------------

            missing_fields = (
                self.REQUIRED_FIELDS
                - set(record.keys())
            )

            for field in sorted(missing_fields):

                schema_errors.append(
                    self.record_reference(
                        record,
                        reason=f"missing_field:{field}",
                    )
                )

            # --------------------------------------------------------
            # Source file
            # --------------------------------------------------------

            source_file = record.get("source_file")

            if not isinstance(source_file, str):
                source_file = "<UNKNOWN>"

            messages_by_source[source_file] += 1

            # --------------------------------------------------------
            # Text
            #
            # We resolve this early because system-event classification
            # depends on the text.
            # --------------------------------------------------------

            text = record.get("text")

            if not isinstance(text, str):

                schema_errors.append(
                    self.record_reference(
                        record,
                        reason="text_is_not_string",
                    )
                )

                text = ""

            # --------------------------------------------------------
            # 2. System event classification
            # --------------------------------------------------------

            system_event_type = self.classify_system_event(text)

            if system_event_type is not None:

                system_event_counts[
                    system_event_type
                ] += 1

                system_events.append(
                    self.record_reference(
                        record,
                        reason=f"system_event:{system_event_type}",
                        extra={
                            "system_event_type": system_event_type,
                            "text": text,
                        },
                    )
                )

            # --------------------------------------------------------
            # 3. Parser warnings
            # --------------------------------------------------------

            warnings = record.get(
                "parse_warnings",
                [],
            )

            if warnings:

                if isinstance(warnings, list):

                    for warning in warnings:

                        warning_text = str(warning)

                        # ------------------------------------------------
                        # Expected system-event warning
                        # ------------------------------------------------
                        #
                        # Only classify the warning as expected when:
                        #
                        #   1. The warning is specifically the known
                        #      sender-delimiter warning.
                        #
                        #   2. The record has no sender.
                        #
                        #   3. The message itself is a recognized
                        #      WhatsApp system event.
                        #
                        # This prevents us from accidentally hiding a
                        # genuine parser problem on a normal message.
                        # ------------------------------------------------

                        sender = record.get("sender")

                        sender_missing = (
                            sender is None
                            or str(sender).strip() == ""
                        )

                        if (
                            warning_text
                            == EXPECTED_SYSTEM_EVENT_WARNING
                            and sender_missing
                            and system_event_type is not None
                        ):

                            expected_parse_warnings.append(
                                self.record_reference(
                                    record,
                                    reason=warning_text,
                                    extra={
                                        "classification": (
                                            "expected_system_event_warning"
                                        ),
                                        "system_event_type": (
                                            system_event_type
                                        ),
                                    },
                                )
                            )

                        else:

                            parse_errors.append(
                                self.record_reference(
                                    record,
                                    reason=warning_text,
                                )
                            )

                else:

                    parse_errors.append(
                        self.record_reference(
                            record,
                            reason="parse_warnings_is_not_a_list",
                        )
                    )

            # --------------------------------------------------------
            # 4. Timestamp
            # --------------------------------------------------------

            timestamp = record.get("timestamp")

            if timestamp is None or timestamp == "":

                missing_timestamps.append(
                    self.record_reference(
                        record,
                        reason="missing_timestamp",
                    )
                )

                parsed_timestamp = None

            else:

                parsed_timestamp = self.parse_timestamp(
                    timestamp
                )

                if parsed_timestamp is None:

                    timestamp_anomalies.append(
                        self.record_reference(
                            record,
                            reason="invalid_timestamp",
                        )
                    )

            timestamps_by_source[source_file].append(
                (
                    str(record.get("message_id", "")),
                    parsed_timestamp,
                )
            )

            # --------------------------------------------------------
            # 5. Sender
            # --------------------------------------------------------

            sender = record.get("sender")

            if (
                sender is None
                or str(sender).strip() == ""
            ):

                # Any recognized system event is expected to have no
                # sender in the exported TXT structure.
                #
                # This includes the encryption preamble, calls,
                # system notifications, and group-management events.

                if system_event_type is not None:

                    pass

                else:

                    missing_senders.append(
                        self.record_reference(
                            record,
                            reason="missing_sender",
                        )
                    )

            else:

                messages_by_sender[str(sender)] += 1

            # --------------------------------------------------------
            # 6. Empty message
            # --------------------------------------------------------

            if not text.strip():

                empty_messages.append(
                    self.record_reference(
                        record,
                        reason="empty_or_whitespace_only_message",
                    )
                )

            # --------------------------------------------------------
            # 7. Message type
            # --------------------------------------------------------

            message_type = record.get(
                "message_type"
            )

            if not isinstance(
                message_type,
                str,
            ):

                schema_errors.append(
                    self.record_reference(
                        record,
                        reason="message_type_is_not_string",
                    )
                )

                message_type = "<UNKNOWN>"

            messages_by_type[message_type] += 1

            if message_type not in KNOWN_MESSAGE_TYPES:

                unexpected_types.append(
                    self.record_reference(
                        record,
                        reason=(
                            "unexpected_message_type:"
                            f"{message_type}"
                        ),
                    )
                )

            # --------------------------------------------------------
            # 8. Large message
            # --------------------------------------------------------

            if len(text) > self.large_message_threshold:

                suspicious_large_messages.append(
                    self.record_reference(
                        record,
                        reason=(
                            "message_exceeds_large_message_threshold"
                        ),
                        extra={
                            "character_count": len(text),
                            "threshold": (
                                self.large_message_threshold
                            ),
                        },
                    )
                )

            # --------------------------------------------------------
            # 9. Boolean fields
            # --------------------------------------------------------

            for field in self.BOOLEAN_FIELDS:

                if field not in record:
                    continue

                if not isinstance(
                    record[field],
                    bool,
                ):

                    schema_errors.append(
                        self.record_reference(
                            record,
                            reason=f"{field}_is_not_boolean",
                        )
                    )

            # --------------------------------------------------------
            # 10. Message ID
            # --------------------------------------------------------

            message_id = record.get("message_id")

            if not isinstance(
                message_id,
                str,
            ):

                schema_errors.append(
                    self.record_reference(
                        record,
                        reason="message_id_is_not_string",
                    )
                )

            elif not message_id:

                schema_errors.append(
                    self.record_reference(
                        record,
                        reason="message_id_is_empty",
                    )
                )

            # --------------------------------------------------------
            # 11. Provenance
            # --------------------------------------------------------

            self.validate_provenance(
                record=record,
                previous_record_by_source=(
                    previous_record_by_source
                ),
                provenance_errors=provenance_errors,
            )

            previous_record_by_source[
                source_file
            ] = record

            # --------------------------------------------------------
            # 12. Exact duplicate key
            #
            # We include provenance because two identical messages
            # legitimately sent at different locations are not the
            # same canonical record.
            # --------------------------------------------------------

            exact_key = (
                source_file,
                record.get("message_id"),
                record.get("timestamp"),
                record.get("sender"),
                text,
                record.get("message_type"),
                record.get("raw_line_start"),
                record.get("raw_line_end"),
            )

            exact_duplicates[exact_key].append(
                record
            )

            # --------------------------------------------------------
            # 13. Potential duplicate key
            #
            # This intentionally ignores message_id and line numbers.
            #
            # Example:
            # "ok" sent twice by the same person at the same second.
            #
            # That MAY be a duplicate, but it can also be legitimate.
            # Therefore it is only a warning/statistic.
            # --------------------------------------------------------

            potential_key = (
                source_file,
                record.get("timestamp"),
                record.get("sender"),
                text,
                record.get("message_type"),
            )

            potential_duplicates[potential_key].append(
                record
            )

        # ============================================================
        # EXACT DUPLICATES
        # ============================================================

        for _, group in exact_duplicates.items():

            if len(group) <= 1:
                continue

            exact_duplicate_groups.append(
                {
                    "source_file": group[0].get(
                        "source_file"
                    ),
                    "count": len(group),
                    "message_ids": [
                        item.get("message_id")
                        for item in group
                    ],
                }
            )

        # ============================================================
        # POTENTIAL DUPLICATES
        # ============================================================

        for _, group in potential_duplicates.items():

            if len(group) <= 1:
                continue

            potential_duplicate_groups.append(
                {
                    "source_file": group[0].get(
                        "source_file"
                    ),
                    "count": len(group),
                    "message_ids": [
                        item.get("message_id")
                        for item in group
                    ],
                    "timestamp": group[0].get(
                        "timestamp"
                    ),
                    "sender": group[0].get(
                        "sender"
                    ),
                    "message_type": group[0].get(
                        "message_type"
                    ),
                    "text_length": len(
                        str(
                            group[0].get(
                                "text",
                                "",
                            )
                        )
                    ),
                }
            )

        # ============================================================
        # TIMESTAMP ORDER
        # ============================================================

        for source_file, entries in (
            timestamps_by_source.items()
        ):

            previous_timestamp = None
            previous_message_id = None

            for (
                message_id,
                current_timestamp,
            ) in entries:

                if current_timestamp is None:
                    continue

                if (
                    previous_timestamp is not None
                    and current_timestamp < previous_timestamp
                ):

                    timestamp_anomalies.append(
                        {
                            "source_file": source_file,
                            "message_id": message_id,
                            "previous_message_id": (
                                previous_message_id
                            ),
                            "reason": (
                                "timestamp_moves_backward"
                            ),
                            "previous_timestamp": (
                                previous_timestamp.isoformat()
                            ),
                            "timestamp": (
                                current_timestamp.isoformat()
                            ),
                        }
                    )

                previous_timestamp = current_timestamp
                previous_message_id = message_id

        # ============================================================
        # EXPECTED WHATSAPP SYSTEM PREAMBLES
        # ============================================================

        expected_preamble_count = sum(
            1
            for record in records
            if self.is_encryption_preamble(
                record.get("text", "")
            )
        )

        # ============================================================
        # PARSER WARNING COUNTS
        # ============================================================

        total_parse_warning_count = (
            len(parse_errors)
            + len(expected_parse_warnings)
        )

        genuine_parse_error_count = len(
            parse_errors
        )

        expected_system_event_warning_count = len(
            expected_parse_warnings
        )

        # ============================================================
        # FINAL STATUS
        # ============================================================

        hard_error_count = (
            len(malformed_records)
            + len(schema_errors)
            + len(timestamp_anomalies)
            + len(provenance_errors)
        )

        if hard_error_count > 0:

            status = "FAIL"

        elif (
            len(parse_errors)
            + len(missing_senders)
            + len(unexpected_types)
            > 0
        ):

            status = "WARN"

        else:

            status = "PASS"

        # ============================================================
        # REPORT
        # ============================================================

        return {
            "schema_version": "1.0",

            "validation_version": "t004.v3",

            "status": status,

            "generated_at": (
                datetime.now()
                .astimezone()
                .isoformat(
                    timespec="seconds"
                )
            ),

            # --------------------------------------------------------
            # Dataset metadata
            # --------------------------------------------------------

            "dataset": {
                "messages_file": str(
                    messages_path
                ),
                "total_messages": len(
                    records
                ),
                "source_files": len(
                    messages_by_source
                ),
                "total_senders": len(
                    messages_by_sender
                ),
            },

            # --------------------------------------------------------
            # Counts
            # --------------------------------------------------------

            "counts": {

                "total_messages": len(
                    records
                ),

                # Genuine parser errors only.
                #
                # Expected system-event warnings are deliberately
                # excluded from this count.
                "parse_error_count": (
                    genuine_parse_error_count
                ),

                # Total parser warnings, including expected system
                # event warnings.
                "total_parse_warning_count": (
                    total_parse_warning_count
                ),

                # Warnings caused by legitimate WhatsApp system
                # generated messages.
                "expected_system_event_warning_count": (
                    expected_system_event_warning_count
                ),

                "malformed_record_count": len(
                    malformed_records
                ),

                "schema_error_count": len(
                    schema_errors
                ),

                "missing_timestamp_count": len(
                    missing_timestamps
                ),

                "missing_sender_count": len(
                    missing_senders
                ),

                "system_event_count": len(
                    system_events
                ),

                "expected_system_preamble_count": (
                    expected_preamble_count
                ),

                "empty_message_count": len(
                    empty_messages
                ),

                "exact_duplicate_group_count": len(
                    exact_duplicate_groups
                ),

                "potential_duplicate_group_count": len(
                    potential_duplicate_groups
                ),

                "suspicious_large_message_count": len(
                    suspicious_large_messages
                ),

                "unexpected_message_type_count": len(
                    unexpected_types
                ),

                "timestamp_anomaly_count": len(
                    timestamp_anomalies
                ),

                "provenance_error_count": len(
                    provenance_errors
                ),
            },

            # --------------------------------------------------------
            # Sender statistics
            # --------------------------------------------------------

            "messages_by_sender": dict(
                sorted(
                    messages_by_sender.items()
                )
            ),

            # --------------------------------------------------------
            # Message type statistics
            # --------------------------------------------------------

            "messages_by_type": dict(
                sorted(
                    messages_by_type.items()
                )
            ),

            # --------------------------------------------------------
            # Source statistics
            # --------------------------------------------------------

            "messages_by_source_file": dict(
                sorted(
                    messages_by_source.items()
                )
            ),

            # --------------------------------------------------------
            # System event statistics
            # --------------------------------------------------------

            "system_event_counts": dict(
                sorted(
                    system_event_counts.items()
                )
            ),

            # --------------------------------------------------------
            # Detailed issues
            # --------------------------------------------------------

            "issues": {

                # Genuine parser warnings/errors only.
                "parse_errors": parse_errors,

                # Expected parser warnings caused by legitimate
                # WhatsApp system events.
                "expected_parse_warnings": (
                    expected_parse_warnings
                ),

                "malformed_records": (
                    malformed_records
                ),

                "schema_errors": (
                    schema_errors
                ),

                "missing_timestamps": (
                    missing_timestamps
                ),

                "missing_senders": (
                    missing_senders
                ),

                "system_events": (
                    system_events
                ),

                "empty_messages": (
                    empty_messages
                ),

                "exact_duplicates": (
                    exact_duplicate_groups
                ),

                "potential_duplicates": (
                    potential_duplicate_groups
                ),

                "suspicious_large_messages": (
                    suspicious_large_messages
                ),

                "unexpected_message_types": (
                    unexpected_types
                ),

                "timestamp_anomalies": (
                    timestamp_anomalies
                ),

                "provenance_errors": (
                    provenance_errors
                ),
            },

            # --------------------------------------------------------
            # Validator configuration
            # --------------------------------------------------------

            "configuration": {

                "large_message_threshold": (
                    self.large_message_threshold
                ),

                "known_message_types": sorted(
                    KNOWN_MESSAGE_TYPES
                ),

                "expected_system_event_warning": (
                    EXPECTED_SYSTEM_EVENT_WARNING
                ),
            },
        }

    # ================================================================
    # HELPERS
    # ================================================================

    @staticmethod
    def parse_timestamp(
        value: Any,
    ) -> datetime | None:

        if not isinstance(
            value,
            str,
        ):
            return None

        value = value.strip()

        if not value:
            return None

        try:

            return datetime.fromisoformat(
                value
            )

        except ValueError:

            return None

    # ----------------------------------------------------------------
    # Encryption preamble
    # ----------------------------------------------------------------

    @staticmethod
    def is_encryption_preamble(
        text: Any,
    ) -> bool:

        if not isinstance(
            text,
            str,
        ):
            return False

        return (
            text.strip()
            == ENCRYPTION_PREAMBLE
        )

    # ----------------------------------------------------------------
    # System-event classifier
    # ----------------------------------------------------------------

    @staticmethod
    def classify_system_event(
        text: Any,
    ) -> str | None:
        """
        Classify WhatsApp-generated messages that legitimately have
        no sender in the TXT export.

        Important:
        This classifier must be conservative.

        A normal conversation message can contain words such as
        "added", "removed", "left", or "made", so we must NOT classify
        a message as a system event merely because those words occur
        somewhere in its text.

        Returns:
            "encryption_notice"
            "call_event"
            "system_notification"
            "group_management_event"
            None
        """

        if not isinstance(text, str):
            return None

        normalized = text.strip()

        # ================================================================
        # 1. Encryption notice
        # ================================================================

        if normalized == ENCRYPTION_PREAMBLE:
            return "encryption_notice"

        # ================================================================
        # 2. Call event
        # ================================================================
        #
        # Actual records observed:
        #
        #     - [Call]
        #
        # Accept the non-dashed form as well for robustness.
        # ================================================================

        if normalized in {
            "[Call]",
            "- [Call]",
        }:
            return "call_event"

        # ================================================================
        # 3. System notification
        # ================================================================
        #
        # Actual records observed:
        #
        #     - [System notification]
        #
        # Only classify the complete system-notification marker.
        # ================================================================

        if normalized in {
            "[System notification]",
            "- [System notification]",
        }:
            return "system_notification"

        # ================================================================
        # 4. Group-management events
        # ================================================================
        #
        # WhatsApp system-generated group events in this dataset have
        # the following important property:
        #
        #     they begin with "- "
        #
        # This is critical because ordinary conversation messages can
        # contain words such as "added", "removed", "left", etc.
        #
        # We therefore NEVER classify a normal message containing one
        # of those words as a group event.
        # ================================================================

        if not normalized.startswith("- "):
            return None

        event_text = normalized[2:].strip()

        if not event_text:
            return None

        # ------------------------------------------------------------
        # Group creation
        #
        # Example:
        #
        #   - User B created the group
        # ------------------------------------------------------------

        if event_text.endswith(" created the group"):
            return "group_management_event"

        # ------------------------------------------------------------
        # Group member added
        #
        # Examples:
        #
        #   - User B added User C and User A
        #   - Someone added User C
        #   - You added User A
        # ------------------------------------------------------------

        if " added " in event_text:
            return "group_management_event"

        # ------------------------------------------------------------
        # Group member removed
        #
        # Examples:
        #
        #   - User A removed User C
        #   - User B removed User D(...)
        # ------------------------------------------------------------

        if " removed " in event_text:
            return "group_management_event"

        # ------------------------------------------------------------
        # Member left
        #
        # Examples:
        #
        #   - User A left
        #   - User B left
        #   - User C left
        # ------------------------------------------------------------

        if event_text.endswith(" left"):
            return "group_management_event"

        # ------------------------------------------------------------
        # Group name changed
        #
        # Examples:
        #
        #   - User B changed the group name to 'Group Alpha'
        #   - You changed the group name to 'Group Beta'
        # ------------------------------------------------------------

        if " changed the group name to " in event_text:
            return "group_management_event"

        # ------------------------------------------------------------
        # Group icon changed
        #
        # Examples:
        #
        #   - User A changed the group icon
        #   - You changed the group icon
        # ------------------------------------------------------------

        if event_text.endswith(" changed the group icon"):
            return "group_management_event"

        # ------------------------------------------------------------
        # Admin-related system events
        #
        # Examples:
        #
        #   - You made You an admin
        #
        # Keep these conservative and require the exact phrase
        # structure rather than matching "made" anywhere.
        # ------------------------------------------------------------

        if event_text.startswith("You made ") and event_text.endswith(
            " an admin"
        ):
            return "group_management_event"

        return None

    # ----------------------------------------------------------------
    # Record reference
    # ----------------------------------------------------------------

    @staticmethod
    def record_reference(
        record: dict[str, Any],
        reason: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        result = {
            "message_id": record.get(
                "message_id"
            ),
            "source_file": record.get(
                "source_file"
            ),
            "raw_line_start": record.get(
                "raw_line_start"
            ),
            "raw_line_end": record.get(
                "raw_line_end"
            ),
            "jsonl_line": record.get(
                "_jsonl_line"
            ),
            "reason": reason,
        }

        if extra:
            result.update(extra)

        return result

    # ----------------------------------------------------------------
    # Provenance validation
    # ----------------------------------------------------------------

    @staticmethod
    def validate_provenance(
        record: dict[str, Any],
        previous_record_by_source: dict[
            str,
            dict[str, Any],
        ],
        provenance_errors: list[dict[str, Any]],
    ) -> None:

        source_file = record.get(
            "source_file"
        )

        start = record.get(
            "raw_line_start"
        )

        end = record.get(
            "raw_line_end"
        )

        # ------------------------------------------------------------
        # Start line
        # ------------------------------------------------------------

        if not isinstance(
            start,
            int,
        ):

            provenance_errors.append(
                DatasetValidator.record_reference(
                    record,
                    reason=(
                        "raw_line_start_is_not_integer"
                    ),
                )
            )

            return

        # ------------------------------------------------------------
        # End line
        # ------------------------------------------------------------

        if not isinstance(
            end,
            int,
        ):

            provenance_errors.append(
                DatasetValidator.record_reference(
                    record,
                    reason=(
                        "raw_line_end_is_not_integer"
                    ),
                )
            )

            return

        # ------------------------------------------------------------
        # Positive lines
        # ------------------------------------------------------------

        if start <= 0:

            provenance_errors.append(
                DatasetValidator.record_reference(
                    record,
                    reason=(
                        "raw_line_start_must_be_positive"
                    ),
                )
            )

        # ------------------------------------------------------------
        # Range direction
        # ------------------------------------------------------------

        if end < start:

            provenance_errors.append(
                DatasetValidator.record_reference(
                    record,
                    reason=(
                        "raw_line_end_before_start"
                    ),
                )
            )

        # ------------------------------------------------------------
        # Compare against previous record from same source.
        # ------------------------------------------------------------

        previous = previous_record_by_source.get(
            source_file
        )

        if previous is None:
            return

        previous_end = previous.get(
            "raw_line_end"
        )

        if (
            isinstance(
                previous_end,
                int,
            )
            and start <= previous_end
        ):

            provenance_errors.append(
                DatasetValidator.record_reference(
                    record,
                    reason=(
                        "raw_line_range_overlaps_previous_message"
                    ),
                    extra={
                        "previous_message_id": (
                            previous.get(
                                "message_id"
                            )
                        ),
                        "previous_raw_line_end": (
                            previous_end
                        ),
                    },
                )
            )


# ====================================================================
# CONVENIENCE FUNCTION
# ====================================================================

def validate_dataset(
    messages_path: str | Path,
    output_path: str | Path,
    large_message_threshold: int = LARGE_MESSAGE_THRESHOLD,
) -> dict[str, Any]:

    messages_path = Path(
        messages_path
    )

    output_path = Path(
        output_path
    )

    validator = DatasetValidator(
        large_message_threshold=(
            large_message_threshold
        )
    )

    report = validator.validate_file(
        messages_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return report


# ====================================================================
# CLI
# ====================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Validate the canonical T003 "
            "WhatsApp dataset."
        )
    )

    parser.add_argument(
        "messages",
        type=Path,
        help=(
            "Path to messages.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/"
            "validation_report.json"
        ),
        help=(
            "Path for validation_report.json"
        ),
    )

    parser.add_argument(
        "--large-message-threshold",
        type=int,
        default=LARGE_MESSAGE_THRESHOLD,
        help=(
            "Character count above which "
            "a message is considered suspicious."
        ),
    )

    args = parser.parse_args()

    report = validate_dataset(
        messages_path=args.messages,
        output_path=args.output,
        large_message_threshold=(
            args.large_message_threshold
        ),
    )

    print(
        json.dumps(
            {
                "status": report["status"],
                **report["counts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()