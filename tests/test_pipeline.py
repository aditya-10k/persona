"""
Unit tests for PersonaPipeline orchestrator and CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.pipeline import PersonaPipeline, PersonaPipelineConfig


@pytest.fixture
def mock_work_dir(tmp_path: Path) -> Path:
    work_dir = tmp_path / "processed"
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def test_pipeline_config_defaults():
    config = PersonaPipelineConfig()
    assert config.version == "1.0.0"
    assert config.target_speaker is None
    assert config.force is False
    assert config.run_eval is False
    assert config.device == "cpu"


def test_resolve_target_speaker_explicit(tmp_path: Path):
    pipeline = PersonaPipeline(
        PersonaPipelineConfig(
            input_path=tmp_path / "raw",
            work_dir=tmp_path / "processed",
            output_dir=tmp_path / "output",
            target_speaker="CustomUser",
        )
    )
    resolved = pipeline.resolve_target_speaker(tmp_path / "nonexistent.json")
    assert resolved == "CustomUser"


def test_resolve_target_speaker_auto_you(tmp_path: Path):
    stats_file = tmp_path / "sender_stats.json"
    stats_file.write_text(json.dumps({
        "senders": {
            "FriendA": {"message_count": 50, "source_files": {"chat1.txt": 50}},
            "You": {"message_count": 500, "source_files": {"chat1.txt": 250, "chat2.txt": 250}},
        }
    }), encoding="utf-8")

    pipeline = PersonaPipeline(
        PersonaPipelineConfig(
            input_path=tmp_path / "raw",
            work_dir=tmp_path / "processed",
            output_dir=tmp_path / "output",
        )
    )
    resolved = pipeline.resolve_target_speaker(stats_file)
    assert resolved == "You"


def test_resolve_target_speaker_auto_other_speaker(tmp_path: Path):
    stats_file = tmp_path / "sender_stats.json"
    stats_file.write_text(json.dumps({
        "senders": {
            "- Someone changed settings": {"message_count": 10, "source_files": {"chat1.txt": 10}},
            "Alice": {"message_count": 300, "source_files": {"chat1.txt": 100, "chat2.txt": 100, "chat3.txt": 100}},
            "Bob": {"message_count": 100, "source_files": {"chat1.txt": 100}},
        }
    }), encoding="utf-8")

    pipeline = PersonaPipeline(
        PersonaPipelineConfig(
            input_path=tmp_path / "raw",
            work_dir=tmp_path / "processed",
            output_dir=tmp_path / "output",
        )
    )
    resolved = pipeline.resolve_target_speaker(stats_file)
    assert resolved == "Alice"


def test_pipeline_directory_initialization(tmp_path: Path):
    work_dir = tmp_path / "processed"
    output_dir = tmp_path / "output"
    pipeline = PersonaPipeline(
        PersonaPipelineConfig(
            input_path=tmp_path / "raw",
            work_dir=work_dir,
            output_dir=output_dir,
        )
    )
    pipeline._ensure_dirs()
    assert work_dir.exists()
    assert output_dir.exists()
    assert (work_dir / "embeddings").exists()
    assert (work_dir / "clustering").exists()
    assert (work_dir / "topics").exists()
    assert (work_dir / "linguistics").exists()
    assert (work_dir / "syntax").exists()
    assert (work_dir / "language").exists()
    assert (work_dir / "discourse").exists()
    assert (work_dir / "situations").exists()
    assert (work_dir / "behavior").exists()
    assert (work_dir / "style").exists()
    assert (work_dir / "inference").exists()
    assert (work_dir / "evidence").exists()
    assert (work_dir / "style_index").exists()
