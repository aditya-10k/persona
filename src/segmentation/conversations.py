"""
Stage T007: Temporal Conversation Segmentation

Segments a flat chronological stream of messages into discrete, bounded
conversational sessions using an inactivity threshold (delta t).
Reconstructs conversational turns (grouping consecutive bursts from the same sender).
"""

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


class ConversationSegmenter:
    """
    Groups messages into discrete conversational sessions based on temporal gaps
    and source file boundaries.
    """

    def __init__(self, gap_hours: float = 4.0):
        self.gap_hours = gap_hours
        self.gap_seconds = gap_hours * 3600.0

    def group_into_turns(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Groups consecutive messages from the same sender into a single conversational turn.
        """
        turns: List[Dict[str, Any]] = []
        current_speaker: Optional[str] = None
        current_turn_messages: List[Dict[str, Any]] = []

        for msg in messages:
            speaker = msg.get("sender")

            if current_speaker is None:
                current_speaker = speaker
                current_turn_messages = [msg]
            elif speaker == current_speaker:
                current_turn_messages.append(msg)
            else:
                # End previous turn
                turn_idx = len(turns) + 1
                turns.append({
                    "turn_id": turn_idx,
                    "speaker": current_speaker,
                    "start_timestamp": current_turn_messages[0]["timestamp"],
                    "end_timestamp": current_turn_messages[-1]["timestamp"],
                    "message_count": len(current_turn_messages),
                    "message_ids": [m["message_id"] for m in current_turn_messages],
                })
                current_speaker = speaker
                current_turn_messages = [msg]

        if current_turn_messages:
            turn_idx = len(turns) + 1
            turns.append({
                "turn_id": turn_idx,
                "speaker": current_speaker,
                "start_timestamp": current_turn_messages[0]["timestamp"],
                "end_timestamp": current_turn_messages[-1]["timestamp"],
                "message_count": len(current_turn_messages),
                "message_ids": [m["message_id"] for m in current_turn_messages],
            })

        return turns

    def build_conversation(
        self, source_file: str, session_index: int, session_messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Constructs a structured conversation session record from a list of messages.
        """
        if not session_messages:
            raise ValueError("Cannot build a conversation from empty message list")

        conv_id = f"{source_file}:c{session_index:04d}"
        start_ts = session_messages[0]["timestamp"]
        end_ts = session_messages[-1]["timestamp"]

        start_dt = datetime.fromisoformat(start_ts)
        end_dt = datetime.fromisoformat(end_ts)
        duration_seconds = max(0, int((end_dt - start_dt).total_seconds()))

        # Determine participants and counts
        participant_counts = Counter()
        initiator = None

        for msg in session_messages:
            sender = msg.get("sender")
            if sender:
                participant_counts[sender] += 1
                if initiator is None:
                    initiator = sender

        turns = self.group_into_turns(session_messages)

        return {
            "conversation_id": conv_id,
            "source_file": source_file,
            "session_index": session_index,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "duration_seconds": duration_seconds,
            "duration_minutes": round(duration_seconds / 60.0, 2),
            "message_count": len(session_messages),
            "turn_count": len(turns),
            "initiator": initiator,
            "participants": sorted(list(participant_counts.keys())),
            "participant_message_counts": dict(participant_counts),
            "turns": turns,
            "messages": session_messages,
        }

    def segment_messages(
        self, messages_iterable: Any
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Generator that streams messages, groups them by temporal gaps (delta t >= gap_seconds)
        and file boundaries, yielding complete conversation session objects.
        """
        current_file: Optional[str] = None
        current_session_messages: List[Dict[str, Any]] = []
        last_dt: Optional[datetime] = None
        session_counter = 0

        for msg in messages_iterable:
            source_file = msg.get("source_file", "unknown")
            msg_dt = datetime.fromisoformat(msg["timestamp"])

            # Check if boundary triggered by new file
            new_file_boundary = (current_file is not None and source_file != current_file)

            # Check if boundary triggered by time gap
            time_gap_boundary = False
            if last_dt is not None and not new_file_boundary:
                time_delta = (msg_dt - last_dt).total_seconds()
                if time_delta >= self.gap_seconds:
                    time_gap_boundary = True

            if new_file_boundary or time_gap_boundary:
                if current_session_messages:
                    session_counter += 1
                    yield self.build_conversation(
                        current_file or "unknown",
                        session_counter,
                        current_session_messages,
                    )
                current_session_messages = []

                if new_file_boundary:
                    session_counter = 0

            current_file = source_file
            last_dt = msg_dt
            current_session_messages.append(msg)

        # Flush final session
        if current_session_messages:
            session_counter += 1
            yield self.build_conversation(
                current_file or "unknown",
                session_counter,
                current_session_messages,
            )


def segment_dataset(
    input_jsonl_path: str,
    output_jsonl_path: str,
    report_output_path: Optional[str] = None,
    gap_hours: float = 4.0,
) -> Dict[str, Any]:
    """
    Streams cleaned_messages.jsonl, segments them into conversations,
    and writes conversations.jsonl and segmentation_report.json.
    """
    in_path = Path(input_jsonl_path)
    out_path = Path(output_jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    segmenter = ConversationSegmenter(gap_hours=gap_hours)

    def _message_generator():
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    yield json.loads(line_str)

    total_conversations = 0
    total_messages = 0
    total_turns = 0
    message_counts: List[int] = []
    durations_minutes: List[float] = []

    conversations_by_file = Counter()
    initiator_counter = Counter()

    # Size buckets
    size_buckets = {
        "1_msg": 0,
        "2_to_5_msgs": 0,
        "6_to_20_msgs": 0,
        "21_to_100_msgs": 0,
        "101_plus_msgs": 0,
    }

    # Duration buckets (minutes)
    duration_buckets = {
        "under_5m": 0,
        "5m_to_30m": 0,
        "30m_to_2h": 0,
        "2h_to_6h": 0,
        "6h_plus": 0,
    }

    with open(out_path, "w", encoding="utf-8") as out_f:
        for conv in segmenter.segment_messages(_message_generator()):
            total_conversations += 1
            m_count = conv["message_count"]
            t_count = conv["turn_count"]
            dur_m = conv["duration_minutes"]

            total_messages += m_count
            total_turns += t_count
            message_counts.append(m_count)
            durations_minutes.append(dur_m)

            conversations_by_file[conv["source_file"]] += 1
            if conv["initiator"]:
                initiator_counter[conv["initiator"]] += 1

            # Categorize size
            if m_count == 1:
                size_buckets["1_msg"] += 1
            elif 2 <= m_count <= 5:
                size_buckets["2_to_5_msgs"] += 1
            elif 6 <= m_count <= 20:
                size_buckets["6_to_20_msgs"] += 1
            elif 21 <= m_count <= 100:
                size_buckets["21_to_100_msgs"] += 1
            else:
                size_buckets["101_plus_msgs"] += 1

            # Categorize duration
            if dur_m < 5.0:
                duration_buckets["under_5m"] += 1
            elif dur_m <= 30.0:
                duration_buckets["5m_to_30m"] += 1
            elif dur_m <= 120.0:
                duration_buckets["30m_to_2h"] += 1
            elif dur_m <= 360.0:
                duration_buckets["2h_to_6h"] += 1
            else:
                duration_buckets["6h_plus"] += 1

            out_f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    import statistics

    median_msgs = statistics.median(message_counts) if message_counts else 0
    median_dur = statistics.median(durations_minutes) if durations_minutes else 0

    report = {
        "schema_version": "1.0",
        "task_id": "T007",
        "status": "COMPLETE",
        "parameters": {
            "gap_hours": gap_hours,
            "gap_seconds": gap_hours * 3600.0,
        },
        "total_conversations": total_conversations,
        "total_messages_segmented": total_messages,
        "total_turns": total_turns,
        "averages": {
            "avg_messages_per_conversation": round(total_messages / total_conversations, 2) if total_conversations else 0,
            "median_messages_per_conversation": median_msgs,
            "avg_turns_per_conversation": round(total_turns / total_conversations, 2) if total_conversations else 0,
            "avg_duration_minutes": round(sum(durations_minutes) / total_conversations, 2) if total_conversations else 0,
            "median_duration_minutes": median_dur,
        },
        "conversation_size_distribution": size_buckets,
        "conversation_duration_distribution": duration_buckets,
        "conversations_by_file": dict(conversations_by_file),
        "top_initiators": dict(initiator_counter.most_common(10)),
    }

    if report_output_path:
        rep_path = Path(report_output_path)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rep_path, "w", encoding="utf-8") as rep_f:
            json.dump(report, rep_f, indent=2, ensure_ascii=False)

    return report
