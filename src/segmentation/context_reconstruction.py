"""
Stage T008: Context Reconstruction

Extracts multi-turn context-response pairs from segmented conversations.
Each unit pairs a target response from the persona user ("You") with the preceding
dialogue context (configurable window of k turns, default k=3).
Identifies conversation initiation turns vs response turns and computes response latency.
"""

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


class ContextReconstructor:
    """
    Reconstructs context-response pairs for the target persona speaker from conversation sessions.
    """

    def __init__(self, max_context_turns: int = 3, target_speaker: str = "You"):
        self.max_context_turns = max_context_turns
        self.target_speaker = target_speaker

    def enrich_turns(
        self, turns: List[Dict[str, Any]], messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enriches turns with concatenated text and analysis_text from the underlying messages.
        """
        msg_map = {m["message_id"]: m for m in messages if "message_id" in m}

        enriched_turns: List[Dict[str, Any]] = []
        for t in turns:
            turn_msgs = [msg_map[mid] for mid in t.get("message_ids", []) if mid in msg_map]

            texts: List[str] = []
            analysis_texts: List[str] = []
            raw_texts: List[str] = []

            for m in turn_msgs:
                txt = m.get("text", "")
                texts.append(txt)

                views = m.get("views", {})
                analysis_txt = views.get("analysis_text", txt)
                if analysis_txt:
                    analysis_texts.append(analysis_txt)

                raw_txt = views.get("raw_text", txt)
                raw_texts.append(raw_txt)

            combined_text = "\n".join(texts)
            combined_analysis = " ".join(analysis_texts)
            combined_raw = "\n".join(raw_texts)

            enr = dict(t)
            enr["text"] = combined_text
            enr["analysis_text"] = combined_analysis
            enr["raw_text"] = combined_raw
            enriched_turns.append(enr)

        return enriched_turns

    def extract_pairs_from_conversation(
        self, conversation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extracts all context-response pairs where speaker == target_speaker.
        """
        conv_id = conversation.get("conversation_id", "unknown")
        source_file = conversation.get("source_file", "unknown")
        raw_turns = conversation.get("turns", [])
        messages = conversation.get("messages", [])

        enriched_turns = self.enrich_turns(raw_turns, messages)
        pairs: List[Dict[str, Any]] = []
        pair_counter = 0

        for i, turn in enumerate(enriched_turns):
            if turn.get("speaker") != self.target_speaker:
                continue

            pair_counter += 1
            pair_id = f"{conv_id}:p{pair_counter:04d}"

            if i == 0:
                # User initiated the conversation session
                is_initiation = True
                context_turns: List[Dict[str, Any]] = []
                context_text = ""
                context_analysis_text = ""
                latency_seconds = None
            else:
                is_initiation = False
                start_idx = max(0, i - self.max_context_turns)
                context_turns = enriched_turns[start_idx:i]

                context_lines: List[str] = []
                context_analysis_lines: List[str] = []
                for ct in context_turns:
                    spk = ct.get("speaker", "Unknown")
                    context_lines.append(f"{spk}: {ct['text']}")
                    context_analysis_lines.append(f"{spk}: {ct['analysis_text']}")

                context_text = "\n".join(context_lines)
                context_analysis_text = "\n".join(context_analysis_lines)

                # Calculate latency from end of immediate previous turn to start of target turn
                prev_turn = enriched_turns[i - 1]
                prev_end = datetime.fromisoformat(prev_turn["end_timestamp"])
                curr_start = datetime.fromisoformat(turn["start_timestamp"])
                latency_seconds = max(0.0, round((curr_start - prev_end).total_seconds(), 2))

            pair_record = {
                "pair_id": pair_id,
                "conversation_id": conv_id,
                "source_file": source_file,
                "is_initiation": is_initiation,
                "context_depth": len(context_turns),
                "response_latency_seconds": latency_seconds,
                "context": [
                    {
                        "turn_id": ct["turn_id"],
                        "speaker": ct["speaker"],
                        "start_timestamp": ct["start_timestamp"],
                        "end_timestamp": ct["end_timestamp"],
                        "message_count": ct["message_count"],
                        "text": ct["text"],
                        "analysis_text": ct["analysis_text"],
                    }
                    for ct in context_turns
                ],
                "context_text": context_text,
                "context_analysis_text": context_analysis_text,
                "target_response": {
                    "turn_id": turn["turn_id"],
                    "speaker": turn["speaker"],
                    "start_timestamp": turn["start_timestamp"],
                    "end_timestamp": turn["end_timestamp"],
                    "message_count": turn["message_count"],
                    "text": turn["text"],
                    "analysis_text": turn["analysis_text"],
                },
                "target_text": turn["text"],
                "target_analysis_text": turn["analysis_text"],
            }
            pairs.append(pair_record)

        return pairs


def reconstruct_context_dataset(
    input_conversations_path: str,
    output_pairs_path: str,
    report_output_path: Optional[str] = None,
    max_context_turns: int = 3,
    target_speaker: str = "You",
) -> Dict[str, Any]:
    """
    Streams conversations.jsonl, extracts multi-turn context-response pairs,
    and writes context_pairs.jsonl and context_pairs_report.json.
    """
    in_path = Path(input_conversations_path)
    out_path = Path(output_pairs_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reconstructor = ContextReconstructor(
        max_context_turns=max_context_turns, target_speaker=target_speaker
    )

    total_pairs = 0
    initiation_pairs = 0
    reply_pairs = 0

    depth_counter = Counter()
    latencies: List[float] = []
    pairs_by_file = Counter()

    with open(in_path, "r", encoding="utf-8") as in_f, open(
        out_path, "w", encoding="utf-8"
    ) as out_f:
        for line in in_f:
            line_str = line.strip()
            if not line_str:
                continue

            conv = json.loads(line_str)
            conv_pairs = reconstructor.extract_pairs_from_conversation(conv)

            for p in conv_pairs:
                total_pairs += 1
                out_f.write(json.dumps(p, ensure_ascii=False) + "\n")

                if p["is_initiation"]:
                    initiation_pairs += 1
                else:
                    reply_pairs += 1
                    if p["response_latency_seconds"] is not None:
                        latencies.append(p["response_latency_seconds"])

                depth_counter[p["context_depth"]] += 1
                pairs_by_file[p["source_file"]] += 1

    import statistics

    median_lat = statistics.median(latencies) if latencies else 0
    avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0

    # Latency distribution buckets
    latency_buckets = {
        "instant_under_30s": sum(1 for lat in latencies if lat < 30.0),
        "fast_30s_to_2m": sum(1 for lat in latencies if 30.0 <= lat < 120.0),
        "moderate_2m_to_10m": sum(1 for lat in latencies if 120.0 <= lat < 600.0),
        "delayed_10m_to_1h": sum(1 for lat in latencies if 600.0 <= lat < 3600.0),
        "prolonged_over_1h": sum(1 for lat in latencies if lat >= 3600.0),
    }

    report = {
        "schema_version": "1.0",
        "task_id": "T008",
        "status": "COMPLETE",
        "parameters": {
            "max_context_turns": max_context_turns,
            "target_speaker": target_speaker,
        },
        "total_context_pairs": total_pairs,
        "initiation_pairs": initiation_pairs,
        "reply_pairs": reply_pairs,
        "context_depth_distribution": {
            f"{k}_turns": depth_counter[k] for k in sorted(depth_counter.keys())
        },
        "response_latency_stats": {
            "average_seconds": avg_lat,
            "median_seconds": median_lat,
            "latency_distribution": latency_buckets,
        },
        "pairs_by_source_file": dict(pairs_by_file),
    }

    if report_output_path:
        rep_path = Path(report_output_path)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rep_path, "w", encoding="utf-8") as rep_f:
            json.dump(report, rep_f, indent=2, ensure_ascii=False)

    return report
