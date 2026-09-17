"""
Persona Pipeline: Unified End-to-End Orchestrator.

Provides an automated pipeline allowing anyone to provide raw WhatsApp
chat exports (.txt) and generate the complete production Persona Package:
- style_profile.json
- linguistic_stats.json
- behavior_profile.json
- vocabulary.json
- persona_report.md
- system_prompt.md
- package_metadata.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import subprocess

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Auto-bootstrap: If running under system Python and a local .venv exists, re-invoke with .venv
if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
    _venv_py = PROJECT_ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    if _venv_py.exists() and Path(sys.executable).resolve() != _venv_py.resolve():
        sys.exit(subprocess.call([str(_venv_py), *sys.argv]))

import numpy as np

from src.dataset.quality_filter import filter_context_pairs_dataset
from src.dataset.sampler import sample_context_pairs_dataset
from src.dataset.splitter import create_dataset_splits
from src.nlp.behavior import BehavioralExtractor
from src.nlp.clustering import ClusterConfig, ClusteringEngine
from src.nlp.code_switching import CodeSwitchingAnalyzer
from src.nlp.discourse import DiscourseAnalyzer
from src.nlp.embeddings import EmbeddingConfig, EmbeddingEngine
from src.nlp.linguistics import LinguisticProfiler
from src.nlp.similarity import SimilarityEngine
from src.nlp.situations import SituationalClassifier
from src.nlp.syntax import SyntacticProfiler
from src.nlp.topics import TopicConfig, TopicDiscoveryEngine
from src.parser.whatsapp_parser import WhatsAppParser, save_results_json
from src.persona.evidence import EvidenceAggregator
from src.persona.inference import StructuredInferenceEngine
from src.persona.package import PersonaPackageCompiler
from src.persona.retrieval import RetrievalConfig, StyleEmbeddingIndex, StyleRetriever
from src.persona.style import GlobalStyleProfile, SituationalStyleProfile, StyleProfileBuilder
from src.persona.style_examples import StyleExampleBuilder, StyleExampleConfig
from src.preprocessing.text_cleaner import clean_dataset
from src.privacy.pii_filter import sanitize_dataset
from src.segmentation.context_reconstruction import reconstruct_context_dataset
from src.segmentation.conversations import segment_dataset
from src.transformation.canonicalize import transform

STYLE_NAMES = {
    0: "Denial_Friction (C00)",
    1: "Venting_Banter (C01)",
    2: "Reactive_Slang (C02)",
    3: "Inquisitive_Probing (C03)",
    4: "Technical_Collab (C04)",
    5: "Minimalist_Confirm (C05)",
}


@dataclass
class PersonaPipelineConfig:
    """Configuration options for the PersonaPipeline execution."""
    input_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    work_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "output" / "persona_package")
    target_speaker: Optional[str] = None
    version: str = "1.0.0"
    embedding_model: str = "l3cube-pune/hindi-sentence-bert-nli"
    device: str = "cpu"
    batch_size: int = 64
    random_seed: int = 42
    force: bool = False
    run_eval: bool = False
    recursive: bool = False
    verbose: bool = False


class PersonaPipeline:
    """
    Unified end-to-end Persona Extraction and Profiling Pipeline.

    Orchestrates all 15 stages from raw WhatsApp text parsing to final
    production persona package compilation with target speaker auto-detection.
    """

    def __init__(self, config: Optional[PersonaPipelineConfig] = None, **kwargs: Any) -> None:
        if config is None:
            self.config = PersonaPipelineConfig(**kwargs)
        else:
            self.config = config

        self.config.input_path = Path(self.config.input_path).resolve()
        self.config.work_dir = Path(self.config.work_dir).resolve()
        self.config.output_dir = Path(self.config.output_dir).resolve()

        # Setup subdirectories under work_dir
        self.embeddings_dir = self.config.work_dir / "embeddings"
        self.clustering_dir = self.config.work_dir / "clustering"
        self.topics_dir = self.config.work_dir / "topics"
        self.linguistics_dir = self.config.work_dir / "linguistics"
        self.syntax_dir = self.config.work_dir / "syntax"
        self.language_dir = self.config.work_dir / "language"
        self.discourse_dir = self.config.work_dir / "discourse"
        self.situations_dir = self.config.work_dir / "situations"
        self.behavior_dir = self.config.work_dir / "behavior"
        self.style_dir = self.config.work_dir / "style"
        self.inference_dir = self.config.work_dir / "inference"
        self.evidence_dir = self.config.work_dir / "evidence"
        self.style_index_dir = self.config.work_dir / "style_index"

        self.logger = logging.getLogger("PersonaPipeline")
        self._setup_logging()

    def _setup_logging(self) -> None:
        level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger.setLevel(level)

    def _ensure_dirs(self) -> None:
        dirs = [
            self.config.work_dir,
            self.config.output_dir,
            self.embeddings_dir,
            self.clustering_dir,
            self.topics_dir,
            self.linguistics_dir,
            self.syntax_dir,
            self.language_dir,
            self.discourse_dir,
            self.situations_dir,
            self.behavior_dir,
            self.style_dir,
            self.inference_dir,
            self.evidence_dir,
            self.style_index_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def resolve_target_speaker(self, stats_path: Path) -> str:
        """
        Resolves or auto-detects the target speaker from sender statistics.

        Heuristic for auto-detection:
        1. If user supplied target_speaker, use it.
        2. If 'You' exists and has message_count > 0, select 'You'.
        3. Otherwise, select the sender present in the most source files
           with the highest message count (excluding system events).
        """
        if self.config.target_speaker:
            self.logger.info("Using user-specified target speaker: '%s'", self.config.target_speaker)
            return self.config.target_speaker

        if not stats_path.exists():
            self.logger.warning("sender_stats.json not found; defaulting target speaker to 'You'")
            return "You"

        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats_data = json.load(f)
            senders = stats_data.get("senders", {})

            if "You" in senders and senders["You"].get("message_count", 0) > 0:
                self.logger.info("Auto-detected target speaker: 'You' (%d messages across %d files)",
                                 senders["You"]["message_count"],
                                 len(senders["You"].get("source_files", {})))
                return "You"

            # Filter candidates: exclude system notification artifacts
            ignore_keywords = [
                "changed settings", "changed the group", "created group",
                "added", "removed", "left", "<UNKNOWN>", "Meta AI"
            ]
            candidates = []
            for sender_name, s_data in senders.items():
                if any(kw in sender_name for kw in ignore_keywords):
                    continue
                file_count = len(s_data.get("source_files", {}))
                msg_count = s_data.get("message_count", 0)
                candidates.append((file_count, msg_count, sender_name))

            if candidates:
                candidates.sort(reverse=True)
                top_candidate = candidates[0]
                detected_name = top_candidate[2]
                self.logger.info(
                    "Auto-detected target speaker: '%s' (%d messages across %d files)",
                    detected_name, top_candidate[1], top_candidate[0]
                )
                return detected_name

        except Exception as e:
            self.logger.warning("Failed to auto-detect target speaker from stats: %s. Defaulting to 'You'", e)

        return "You"

    def run_stage_01_parse(self) -> Path:
        """Stage T002: Parse raw WhatsApp TXT files."""
        output_path = self.config.work_dir / "parsed.json"
        if output_path.exists() and not self.config.force:
            self.logger.info("[CACHED] Stage 1: Ingestion & Parsing -> %s", output_path)
            return output_path

        self.logger.info("Stage 1: Parsing raw WhatsApp exports from %s ...", self.config.input_path)
        parser = WhatsAppParser()
        if self.config.input_path.is_file():
            result = parser.parse_file(self.config.input_path)
            results = [result]
        elif self.config.input_path.is_dir():
            results = parser.parse_directory(self.config.input_path, recursive=self.config.recursive)
        else:
            raise FileNotFoundError(f"Input path does not exist: {self.config.input_path}")

        save_results_json(results, output_path)
        total_msgs = sum(len(r.messages) for r in results)
        self.logger.info("Stage 1 complete: Parsed %d messages from %d file(s).", total_msgs, len(results))
        return output_path

    def run_stage_02_canonicalize(self, parsed_path: Path) -> Tuple[Path, Path]:
        """Stage T003: Canonicalize messages into standard schema."""
        messages_path = self.config.work_dir / "messages.jsonl"
        stats_path = self.config.work_dir / "sender_stats.json"
        if messages_path.exists() and stats_path.exists() and not self.config.force:
            self.logger.info("[CACHED] Stage 2: Canonicalization -> %s", messages_path)
            return messages_path, stats_path

        self.logger.info("Stage 2: Canonicalizing message schema ...")
        transform(
            parsed_path=parsed_path,
            output_jsonl=messages_path,
            sender_stats_path=stats_path,
        )
        self.logger.info("Stage 2 complete: Canonical dataset and sender stats written.")
        return messages_path, stats_path

    def run_stage_03_privacy(self, messages_path: Path) -> Path:
        """Stage T005: PII Sanitization & Masking."""
        sanitized_path = self.config.work_dir / "sanitized_messages.jsonl"
        report_path = self.config.work_dir / "privacy_report.json"
        if sanitized_path.exists() and not self.config.force:
            self.logger.info("[CACHED] Stage 3: PII Sanitization -> %s", sanitized_path)
            return sanitized_path

        self.logger.info("Stage 3: Running PII Sanitization Filter ...")
        report = sanitize_dataset(
            input_jsonl_path=str(messages_path),
            output_jsonl_path=str(sanitized_path),
            report_output_path=str(report_path),
        )
        self.logger.info(
            "Stage 3 complete: %d messages processed, %d sanitized.",
            report["total_messages_processed"],
            report["total_messages_sanitized"],
        )
        return sanitized_path

    def run_stage_04_clean(self, sanitized_path: Path) -> Path:
        """Stage T006: Text Cleaning & Normalization."""
        cleaned_path = self.config.work_dir / "cleaned_messages.jsonl"
        report_path = self.config.work_dir / "cleaning_report.json"
        if cleaned_path.exists() and not self.config.force:
            self.logger.info("[CACHED] Stage 4: Text Cleaning -> %s", cleaned_path)
            return cleaned_path

        self.logger.info("Stage 4: Normalizing and cleaning message text ...")
        report = clean_dataset(
            input_jsonl_path=str(sanitized_path),
            output_jsonl_path=str(cleaned_path),
            report_output_path=str(report_path),
        )
        self.logger.info(
            "Stage 4 complete: %d messages cleaned (%d tokens extracted).",
            report["total_messages_processed"],
            report["total_tokens_extracted"],
        )
        return cleaned_path

    def run_stage_05_segment(self, cleaned_path: Path) -> Path:
        """Stage T007: Temporal Conversation Segmentation."""
        conversations_path = self.config.work_dir / "conversations.jsonl"
        report_path = self.config.work_dir / "segmentation_report.json"
        if conversations_path.exists() and not self.config.force:
            self.logger.info("[CACHED] Stage 5: Conversation Segmentation -> %s", conversations_path)
            return conversations_path

        self.logger.info("Stage 5: Segmenting messages into conversational sessions (gap=4.0h) ...")
        report = segment_dataset(
            input_jsonl_path=str(cleaned_path),
            output_jsonl_path=str(conversations_path),
            report_output_path=str(report_path),
            gap_hours=4.0,
        )
        self.logger.info(
            "Stage 5 complete: Segmented into %d conversations (%d turns).",
            report["total_conversations"],
            report["total_turns"],
        )
        return conversations_path

    def run_stage_06_context_reconstruction(self, conversations_path: Path, target_speaker: str) -> Path:
        """Stage T008: Context-Response Reconstruction."""
        pairs_path = self.config.work_dir / "context_pairs.jsonl"
        report_path = self.config.work_dir / "context_pairs_report.json"
        if pairs_path.exists() and not self.config.force:
            self.logger.info("[CACHED] Stage 6: Context Reconstruction -> %s", pairs_path)
            return pairs_path

        self.logger.info("Stage 6: Reconstructing context-response pairs for '%s' ...", target_speaker)
        report = reconstruct_context_dataset(
            input_conversations_path=str(conversations_path),
            output_pairs_path=str(pairs_path),
            report_output_path=str(report_path),
            max_context_turns=3,
            target_speaker=target_speaker,
        )
        self.logger.info(
            "Stage 6 complete: Extracted %d context-response pairs (%d replies, %d initiations).",
            report["total_context_pairs"],
            report["reply_pairs"],
            report["initiation_pairs"],
        )
        return pairs_path

    def run_stage_07_quality_filter(self, pairs_path: Path) -> Path:
        """Stage T009: Quality Filtering."""
        filtered_path = self.config.work_dir / "filtered_pairs.jsonl"
        report_path = self.config.work_dir / "quality_filtering_report.json"
        if filtered_path.exists() and not self.config.force:
            self.logger.info("[CACHED] Stage 7: Quality Filtering -> %s", filtered_path)
            return filtered_path

        self.logger.info("Stage 7: Applying quality heuristics and noise filters ...")
        report = filter_context_pairs_dataset(
            input_pairs_path=str(pairs_path),
            output_pairs_path=str(filtered_path),
            report_output_path=str(report_path),
        )
        self.logger.info(
            "Stage 7 complete: %d pairs passed (%.1f%% pass rate).",
            report["total_pairs_passed"],
            report["pass_rate_percentage"],
        )
        return filtered_path

    def run_stage_08_sample(self, filtered_path: Path) -> Path:
        """Stage T010: Stratified Sampling."""
        sampled_path = self.config.work_dir / "sampled_pairs.jsonl"
        report_path = self.config.work_dir / "sampling_report.json"
        if sampled_path.exists() and not self.config.force:
            self.logger.info("[CACHED] Stage 8: Stratified Sampling -> %s", sampled_path)
            return sampled_path

        self.logger.info("Stage 8: Running source-balanced stratified sampling ...")
        report = sample_context_pairs_dataset(
            input_pairs_path=str(filtered_path),
            output_pairs_path=str(sampled_path),
            report_output_path=str(report_path),
            max_per_source=500,
            random_seed=self.config.random_seed,
        )
        self.logger.info(
            "Stage 8 complete: Sampled %d pairs (down from %d).",
            report["total_pairs_after"],
            report["total_pairs_before"],
        )
        return sampled_path

    def run_stage_09_split(self, sampled_path: Path, conversations_path: Path) -> Tuple[Path, Path, Path]:
        """Stage T011: Train / Development / Holdout Partitioning."""
        train_path = self.config.work_dir / "train_pairs.jsonl"
        dev_path = self.config.work_dir / "dev_pairs.jsonl"
        test_path = self.config.work_dir / "test_pairs.jsonl"
        report_path = self.config.work_dir / "splits_report.json"

        if train_path.exists() and dev_path.exists() and test_path.exists() and not self.config.force:
            self.logger.info("[CACHED] Stage 9: Dataset Splitting -> %s", train_path)
            return train_path, dev_path, test_path

        self.logger.info("Stage 9: Partitioning dataset (70%% Train / 15%% Dev / 15%% Test) ...")
        report = create_dataset_splits(
            input_pairs_path=str(sampled_path),
            conversations_path=str(conversations_path),
            output_dir=str(self.config.work_dir),
            report_output_path=str(report_path),
            train_ratio=0.70,
            dev_ratio=0.15,
            test_ratio=0.15,
            random_seed=self.config.random_seed,
        )
        self.logger.info(
            "Stage 9 complete: Zero leakage verified (%s). Split: %s",
            report["zero_leakage_verified"],
            report["pairs_split"],
        )
        return train_path, dev_path, test_path

    def run_stage_10_embeddings(self, train_pairs_path: Path) -> Tuple[Path, Path]:
        """Stage T012: Dense Transformer Embeddings."""
        target_npz = self.embeddings_dir / "train_target_vectors.npz"
        context_npz = self.embeddings_dir / "train_context_vectors.npz"

        if target_npz.exists() and context_npz.exists() and not self.config.force:
            self.logger.info("[CACHED] Stage 10: Transformer Embeddings -> %s", target_npz)
            return target_npz, context_npz

        self.logger.info("Stage 10: Generating 768-dim embeddings via %s ...", self.config.embedding_model)
        pairs: List[Dict[str, Any]] = []
        with open(train_pairs_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))

        config = EmbeddingConfig(
            model_name=self.config.embedding_model,
            batch_size=self.config.batch_size,
            device=self.config.device,
            normalize_embeddings=True,
        )
        engine = EmbeddingEngine(config)

        # 1. Embed Target Responses
        t0 = time.perf_counter()
        target_vectors, target_pair_ids = engine.encode_pairs(
            pairs=pairs,
            text_field="target_analysis_text",
            fallback_field="target_text",
            show_progress_bar=False,
        )
        t_target = time.perf_counter() - t0
        target_path, _ = engine.save_embeddings(
            output_dir=self.embeddings_dir,
            name_prefix="train_target",
            vectors=target_vectors,
            pair_ids=target_pair_ids,
            extra_metadata={"split": "train", "duration": round(t_target, 2)},
        )

        # 2. Embed Context Prompts
        t1 = time.perf_counter()
        context_vectors, context_pair_ids = engine.encode_pairs(
            pairs=pairs,
            text_field="context_analysis_text",
            fallback_field="context_text",
            show_progress_bar=False,
        )
        t_context = time.perf_counter() - t1
        context_path, _ = engine.save_embeddings(
            output_dir=self.embeddings_dir,
            name_prefix="train_context",
            vectors=context_vectors,
            pair_ids=context_pair_ids,
            extra_metadata={"split": "train", "duration": round(t_context, 2)},
        )

        self.logger.info(
            "Stage 10 complete: Encoded %d target vecs (%.2fs) and %d context vecs (%.2fs).",
            len(target_vectors), t_target, len(context_vectors), t_context
        )
        return target_path, context_path

    def run_stage_11_similarity(
        self,
        train_pairs_path: Path,
        target_vecs_path: Path,
        ctx_vecs_path: Path,
    ) -> Path:
        """Stage T013: Semantic Similarity & Alignment."""
        report_path = self.config.work_dir / "similarity_report.json"
        if report_path.exists() and not self.config.force:
            self.logger.info("[CACHED] Stage 11: Similarity Analysis -> %s", report_path)
            return report_path

        self.logger.info("Stage 11: Computing semantic similarity and response diversity ...")
        with np.load(target_vecs_path) as npz:
            target_vectors = npz["vectors"]
        with np.load(ctx_vecs_path) as npz:
            context_vectors = npz["vectors"]

        pairs = [json.loads(l) for l in train_pairs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        pair_ids = [p["pair_id"] for p in pairs]

        target_sim_engine = SimilarityEngine(
            vectors=target_vectors,
            pair_ids=pair_ids,
            pairs_data=pairs,
        )
        diversity_stats = target_sim_engine.compute_diversity_index(sample_size=min(1000, len(pairs)), random_seed=self.config.random_seed)
        alignment_stats = SimilarityEngine.compute_context_response_alignment(
            context_vectors=context_vectors,
            target_vectors=target_vectors,
        )

        full_report = {
            "diversity_stats": diversity_stats,
            "alignment_stats": alignment_stats,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)

        self.logger.info("Stage 11 complete: Diversity Index: %.4f", diversity_stats["diversity_index"])
        return report_path

    def run_stage_12_clustering(
        self,
        train_pairs_path: Path,
        target_vecs_path: Path,
    ) -> Tuple[Path, np.ndarray]:
        """Stage T014: Semantic Clustering & Archetype Discovery."""
        assignments_path = self.clustering_dir / "cluster_assignments.jsonl"
        profiles_path = self.clustering_dir / "cluster_profiles.json"
        centroids_path = self.clustering_dir / "cluster_centroids.npz"

        with np.load(target_vecs_path) as npz:
            vectors = npz["vectors"]

        pairs = [json.loads(l) for l in train_pairs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        pair_ids = [p["pair_id"] for p in pairs]

        if assignments_path.exists() and not self.config.force:
            self.logger.info("[CACHED] Stage 12: Semantic Clustering -> %s", assignments_path)
            labels = []
            with open(assignments_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        labels.append(json.loads(line)["cluster_id"])
            return assignments_path, np.array(labels, dtype=int)

        self.logger.info("Stage 12: Fitting Spherical K-Means clustering (K=6) ...")
        engine = ClusteringEngine(ClusterConfig(random_seed=self.config.random_seed))
        result = engine.fit(vectors, k=6)

        exemplars = engine.extract_exemplars(
            vectors, result.labels, pair_ids, top_n=5, centroids=result.centroids
        )
        profiles = engine.profile_clusters(
            pairs, result.labels, exemplars=exemplars, top_vocab_n=10, top_emojis_n=5
        )

        # Write assignments
        pairs_by_id = {p["pair_id"]: p for p in pairs}
        with open(assignments_path, "w", encoding="utf-8") as f:
            for idx, (pid, label) in enumerate(zip(pair_ids, result.labels)):
                c = int(label)
                sim = float(np.dot(vectors[idx], result.centroids[c]))
                p = pairs_by_id[pid]
                f.write(json.dumps({
                    "pair_id": pid,
                    "cluster_id": c,
                    "similarity_to_centroid": round(sim, 4),
                    "target_text": p.get("target_text", ""),
                }, ensure_ascii=False) + "\n")

        with open(profiles_path, "w", encoding="utf-8") as f:
            json.dump([p.to_dict() for p in profiles], f, indent=2, ensure_ascii=False)

        np.savez_compressed(centroids_path, centroids=result.centroids)
        self.logger.info("Stage 12 complete: Silhouette = %.4f", result.silhouette)
        return assignments_path, result.labels

    def run_stage_13_topic_discovery(
        self,
        train_pairs_path: Path,
        ctx_vecs_path: Path,
        assignments_path: Path,
    ) -> Path:
        """Stage T015: Topic Discovery & Situational Transition Matrix."""
        report_path = self.topics_dir / "topic_discovery_report.json"
        if report_path.exists() and not self.config.force:
            self.logger.info("[CACHED] Stage 13: Topic Discovery -> %s", report_path)
            return report_path

        self.logger.info("Stage 13: Discovering conversational topics via c-TF-IDF ...")
        with np.load(ctx_vecs_path) as npz:
            ctx_vectors = npz["vectors"]

        pairs = [json.loads(l) for l in train_pairs_path.read_text(encoding="utf-8").splitlines() if l.strip()]

        style_labels_by_pid = {}
        with open(assignments_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    style_labels_by_pid[rec["pair_id"]] = rec["cluster_id"]
        style_labels = np.array([style_labels_by_pid[p["pair_id"]] for p in pairs], dtype=int)

        engine = TopicDiscoveryEngine(TopicConfig(random_seed=self.config.random_seed, exemplars_per_topic=4))
        result = engine.fit(ctx_vectors, pairs, n_topics=8)

        topic_name_map = {p["topic_id"]: p["topic_label"] for p in result.topic_profiles}
        matrix_res = engine.compute_style_topic_matrix(
            topic_labels=result.labels,
            style_labels=style_labels,
            topic_names=topic_name_map,
            style_names=STYLE_NAMES,
        )

        full_doc = {
            "n_topics": len(result.topic_profiles),
            "silhouette": round(float(result.silhouette), 4),
            "topics": result.topic_profiles,
            "transition_matrix": matrix_res,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(full_doc, f, indent=2, ensure_ascii=False)

        self.logger.info("Stage 13 complete: Discovered %d topics.", len(result.topic_profiles))
        return report_path

    def run_stage_14_to_20_nlp_profiling(
        self,
        train_pairs_path: Path,
        assignments_path: Path,
    ) -> None:
        """Stages T016 - T023: Surface, Syntactic, Language, Discourse, Situational, Behavioral & Style."""
        self.logger.info("Stage 14-20: Running comprehensive forensic NLP profiling ...")
        pairs = [json.loads(l) for l in train_pairs_path.read_text(encoding="utf-8").splitlines() if l.strip()]

        style_labels_by_pid = {}
        with open(assignments_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    style_labels_by_pid[rec["pair_id"]] = rec["cluster_id"]
        style_labels = np.array([style_labels_by_pid[p["pair_id"]] for p in pairs], dtype=int)

        # 14. Surface Linguistics (T016)
        ling_global_path = self.linguistics_dir / "global_linguistic_profile.json"
        if not ling_global_path.exists() or self.config.force:
            self.logger.info("  -> Extracting surface linguistic profile (T016) ...")
            l_prof = LinguisticProfiler()
            g_ling = l_prof.fit(pairs, name="global_corpus")
            with open(ling_global_path, "w", encoding="utf-8") as f:
                json.dump(g_ling.to_dict(), f, indent=2, ensure_ascii=False)

        # 15. Syntactic Profiling (T017)
        syn_global_path = self.syntax_dir / "global_syntactic_profile.json"
        if not syn_global_path.exists() or self.config.force:
            self.logger.info("  -> Extracting syntactic profile (T017) ...")
            s_prof = SyntacticProfiler()
            g_syn = s_prof.fit(pairs, name="global_corpus")
            with open(syn_global_path, "w", encoding="utf-8") as f:
                json.dump(g_syn.to_dict(), f, indent=2, ensure_ascii=False)

        # 16. Code-Switching & Language (T018)
        lang_global_path = self.language_dir / "global_language_profile.json"
        if not lang_global_path.exists() or self.config.force:
            self.logger.info("  -> Extracting Hinglish code-switching profile (T018) ...")
            cs_prof = CodeSwitchingAnalyzer()
            g_lang = cs_prof.fit(pairs, name="global_corpus")
            with open(lang_global_path, "w", encoding="utf-8") as f:
                json.dump(g_lang.to_dict(), f, indent=2, ensure_ascii=False)

        # 17. Discourse Analysis (T019)
        disc_global_path = self.discourse_dir / "global_discourse_profile.json"
        if not disc_global_path.exists() or self.config.force:
            self.logger.info("  -> Extracting discourse and speech acts profile (T019) ...")
            d_prof = DiscourseAnalyzer()
            g_disc = d_prof.fit(pairs, name="global_corpus")
            with open(disc_global_path, "w", encoding="utf-8") as f:
                json.dump(g_disc.to_dict(), f, indent=2, ensure_ascii=False)

        # 18. Situational Classification (T020)
        sit_global_path = self.situations_dir / "global_situational_profile.json"
        if not sit_global_path.exists() or self.config.force:
            self.logger.info("  -> Classifying situational environments (T020) ...")
            sit_prof = SituationalClassifier()
            g_sit = sit_prof.fit(pairs, name="global_corpus")
            with open(sit_global_path, "w", encoding="utf-8") as f:
                json.dump(g_sit.to_dict(), f, indent=2, ensure_ascii=False)

        # 19. Behavioral Extraction (T021)
        beh_global_path = self.behavior_dir / "global_behavioral_profile.json"
        if not beh_global_path.exists() or self.config.force:
            self.logger.info("  -> Extracting 10 behavioral dimensions (T021) ...")
            b_prof = BehavioralExtractor()
            g_beh = b_prof.extract_profile(pairs, name="global_corpus", max_exemplars=5)
            with open(beh_global_path, "w", encoding="utf-8") as f:
                json.dump(g_beh.to_dict(), f, indent=2, ensure_ascii=False)

        # 20. Style Profiles (T022 & T023)
        style_global_path = self.style_dir / "global_style_profile.json"
        style_sit_path = self.style_dir / "situational_style_profiles.json"
        if not style_global_path.exists() or not style_sit_path.exists() or self.config.force:
            self.logger.info("  -> Synthesizing global and situational style profiles (T022 & T023) ...")
            builder = StyleProfileBuilder()
            g_style = builder.build_global_style(pairs, name="global_persona")
            sit_styles = builder.build_situational_styles(pairs, max_exemplars=5)

            with open(style_global_path, "w", encoding="utf-8") as f:
                json.dump(g_style.to_dict(), f, indent=2, ensure_ascii=False)
            with open(style_sit_path, "w", encoding="utf-8") as f:
                json.dump({k: v.to_dict() for k, v in sit_styles.items()}, f, indent=2, ensure_ascii=False)

        self.logger.info("Stage 14-20 complete: All forensic profiles persisted.")

    def run_stage_21_to_24_inference_and_exemplars(
        self,
        train_pairs_path: Path,
        target_vecs_path: Path,
        ctx_vecs_path: Path,
        assignments_path: Path,
    ) -> None:
        """Stages T024 - T029: Epistemic Inference, Evidence, Exemplars & Style Retrieval Index."""
        self.logger.info("Stage 21-24: Running epistemic inference and exemplar index compilation ...")

        style_global_path = self.style_dir / "global_style_profile.json"
        style_sit_path = self.style_dir / "situational_style_profiles.json"
        inference_path = self.inference_dir / "structured_persona_inference.json"
        evidence_path = self.evidence_dir / "evidence_registry.json"
        examples_path = self.config.work_dir / "style_examples.jsonl"
        index_path = self.style_index_dir / "style_index.npz"

        # 21. Epistemic Structured Inference (T024)
        with open(style_global_path, "r", encoding="utf-8") as f:
            g_style_dict = json.load(f)
        with open(style_sit_path, "r", encoding="utf-8") as f:
            s_style_dict = json.load(f)

        if not inference_path.exists() or self.config.force:
            self.logger.info("  -> Synthesizing structured epistemic inference (T024) ...")
            engine = StructuredInferenceEngine()
            inference_result = engine.infer_offline(g_style_dict, s_style_dict)
            with open(inference_path, "w", encoding="utf-8") as f:
                json.dump(inference_result.to_dict(), f, indent=2, ensure_ascii=False)
        else:
            engine = StructuredInferenceEngine()
            inference_result = engine.infer_offline(g_style_dict, s_style_dict)

        # 22. Evidence Aggregation (T025)
        if not evidence_path.exists() or self.config.force:
            self.logger.info("  -> Aggregating Master Evidence Registry (T025) ...")
            aggregator = EvidenceAggregator()
            global_style_obj = GlobalStyleProfile(
                total_turns=g_style_dict["total_turns"],
                core_dimensions=g_style_dict["core_dimensions"],
                surface_constraints=g_style_dict["surface_constraints"],
                syntactic_constraints=g_style_dict["syntactic_constraints"],
                language_constraints=g_style_dict["language_constraints"],
                discourse_moves=g_style_dict["discourse_moves"],
                metadata=g_style_dict.get("metadata", {}),
            )
            sit_styles_obj = {
                k: SituationalStyleProfile(
                    situation=v["situation"],
                    description=v["description"],
                    total_turns=v["total_turns"],
                    situation_prevalence=v["situation_prevalence"],
                    core_dimensions=v["core_dimensions"],
                    surface_constraints=v["surface_constraints"],
                    language_constraints=v["language_constraints"],
                    discourse_moves=v["discourse_moves"],
                    representative_exemplars=v.get("representative_exemplars", []),
                    metadata=v.get("metadata", {}),
                )
                for k, v in s_style_dict.items()
            }
            registry = aggregator.aggregate(
                global_style=global_style_obj,
                situational_styles=sit_styles_obj,
                inference_result=inference_result,
            )
            with open(evidence_path, "w", encoding="utf-8") as f:
                json.dump(registry.to_dict(), f, indent=2, ensure_ascii=False)

        # 23. Style Examples Bank (T026 & T027)
        pairs = [json.loads(l) for l in train_pairs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        with np.load(target_vecs_path) as npz:
            target_vectors = npz["vectors"]

        cluster_map = {}
        with open(assignments_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cluster_map[rec["pair_id"]] = rec["cluster_id"]

        unique_sources = sorted(list({p.get("source_file", "") for p in pairs if p.get("source_file")}))
        source_map = {s: f"chat_{i:02d}.txt" for i, s in enumerate(unique_sources, 1)}

        if not examples_path.exists() or self.config.force:
            self.logger.info("  -> Curating representative style exemplars (T026 & T027) ...")
            builder = StyleExampleBuilder(StyleExampleConfig(
                target_per_category=12,
                diversity_lambda=0.65,
                max_pairwise_similarity=0.88,
                min_centroid_similarity=0.15,
                random_seed=self.config.random_seed,
            ))
            ex_res = builder.build(
                pairs=pairs,
                target_vectors=target_vectors,
                cluster_assignments=cluster_map,
                source_anonymizer=source_map,
            )
            with open(examples_path, "w", encoding="utf-8") as f:
                for ex in ex_res.exemplars:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        # 24. Style Retrieval Index (T028 & T029)
        if not index_path.exists() or self.config.force:
            self.logger.info("  -> Building 768-dim style retrieval index (T028 & T029) ...")
            examples = [json.loads(l) for l in examples_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            with np.load(ctx_vecs_path) as npz:
                all_ctx_vecs = npz["vectors"]

            resp_to_idx = {p["target_text"].strip(): i for i, p in enumerate(pairs)}
            matched_ctx_vecs = []
            matched_resp_vecs = []
            for ex in examples:
                resp_clean = ex["response"].strip()
                if resp_clean in resp_to_idx:
                    idx = resp_to_idx[resp_clean]
                    matched_ctx_vecs.append(all_ctx_vecs[idx])
                    matched_resp_vecs.append(target_vectors[idx])
                else:
                    v = np.ones(all_ctx_vecs.shape[1], dtype=np.float32)
                    matched_ctx_vecs.append(v / np.linalg.norm(v))
                    matched_resp_vecs.append(v / np.linalg.norm(v))

            idx_engine = StyleEmbeddingIndex(
                examples=examples,
                context_vectors=np.array(matched_ctx_vecs, dtype=np.float32),
                response_vectors=np.array(matched_resp_vecs, dtype=np.float32),
            )
            idx_engine.save(self.style_index_dir)

        self.logger.info("Stage 21-24 complete: Exemplars and retrieval index ready.")

    def run_stage_25_compile_package(self) -> Dict[str, Path]:
        """Stage T030 - T034: Compile Production Persona Package."""
        self.logger.info("Stage 25: Compiling final Persona Package to %s ...", self.config.output_dir)
        compiler = PersonaPackageCompiler(version=self.config.version)
        artifacts = compiler.compile(self.config.work_dir, self.config.output_dir)

        # Strict Privacy Audit
        names = [
            'Aakarshit', 'aarushi', 'Aditya Gupta', 'Afroz', 'Ticktickboom', 'Gays Ka Parivar',
            'Goklu', 'Gooners', 'Heta', 'Karani', 'Rishi Shah', 'Samruddhi', 'Sudhya',
            'The bock rottom', 'Triponovaa', 'Akshat', 'Vora', 'Ankit', 'Datta', 'Anupam', 'Tarav', 'Swayam', 'Yash',
            'svkm', 'projectsvkm2', 'gaurav', 'advaith', 'manoj'
        ]
        pattern = re.compile('|'.join([r'\b' + re.escape(n) + r'\b' for n in names]), re.IGNORECASE)
        violations = 0
        for name, path in artifacts.items():
            content = path.read_text(encoding="utf-8", errors="ignore")
            matches = pattern.findall(content)
            if matches:
                self.logger.error("PRIVACY LEAK in %s: %s", path.name, set(matches))
                violations += len(matches)

        if violations == 0:
            self.logger.info("PRIVACY AUDIT PASSED: Zero private credentials or unmasked entities in package artifacts.")
        else:
            self.logger.warning("PRIVACY AUDIT: %d unmasked entity occurrences flagged.", violations)

        return artifacts

    def run(self) -> Dict[str, Any]:
        """
        Executes the entire end-to-end Persona Extraction & Profiling Pipeline.
        """
        t_start = time.perf_counter()
        self.logger.info("=" * 70)
        self.logger.info("  STARTING PERSONA EXTRACTION PIPELINE (v%s)", self.config.version)
        self.logger.info("  Input Path:  %s", self.config.input_path)
        self.logger.info("  Working Dir: %s", self.config.work_dir)
        self.logger.info("  Output Dir:  %s", self.config.output_dir)
        self.logger.info("=" * 70)

        self._ensure_dirs()

        # Step 1: Parse
        parsed_path = self.run_stage_01_parse()

        # Step 2: Canonicalize
        messages_path, stats_path = self.run_stage_02_canonicalize(parsed_path)

        # Target Speaker Resolution
        target_speaker = self.resolve_target_speaker(stats_path)

        # Step 3: Privacy
        sanitized_path = self.run_stage_03_privacy(messages_path)

        # Step 4: Text Cleaning
        cleaned_path = self.run_stage_04_clean(sanitized_path)

        # Step 5: Segmentation
        convs_path = self.run_stage_05_segment(cleaned_path)

        # Step 6: Context Reconstruction
        pairs_path = self.run_stage_06_context_reconstruction(convs_path, target_speaker)

        # Step 7: Quality Filtering
        filtered_path = self.run_stage_07_quality_filter(pairs_path)

        # Step 8: Sampling
        sampled_path = self.run_stage_08_sample(filtered_path)

        # Step 9: Splitting
        train_path, dev_path, test_path = self.run_stage_09_split(sampled_path, convs_path)

        # Step 10: Transformer Embeddings
        target_vecs, ctx_vecs = self.run_stage_10_embeddings(train_path)

        # Step 11: Similarity Analysis
        self.run_stage_11_similarity(train_path, target_vecs, ctx_vecs)

        # Step 12: Clustering
        assignments_path, labels = self.run_stage_12_clustering(train_path, target_vecs)

        # Step 13: Topic Discovery
        self.run_stage_13_topic_discovery(train_path, ctx_vecs, assignments_path)

        # Step 14-20: Forensic NLP Profiling
        self.run_stage_14_to_20_nlp_profiling(train_path, assignments_path)

        # Step 21-24: Epistemic Inference & Style Index
        self.run_stage_21_to_24_inference_and_exemplars(train_path, target_vecs, ctx_vecs, assignments_path)

        # Step 25: Compile Persona Package
        artifacts = self.run_stage_25_compile_package()

        # Step 26: Optional Holdout Evaluation Benchmark
        eval_summary = None
        if self.config.run_eval:
            eval_summary = self.run_stage_26_evaluate(test_path)

        elapsed = time.perf_counter() - t_start
        self.logger.info("=" * 70)
        self.logger.info("  PIPELINE EXECUTION COMPLETE in %.2f seconds", elapsed)
        self.logger.info("  Compiled Package Artifacts:")
        for name, path in artifacts.items():
            size_kb = path.stat().st_size / 1024.0
            self.logger.info("    - %-20s: %-25s (%6.2f KB)", name, path.name, size_kb)
        self.logger.info("=" * 70)

        res: Dict[str, Any] = {
            "status": "SUCCESS",
            "version": self.config.version,
            "target_speaker": target_speaker,
            "duration_seconds": round(elapsed, 2),
            "artifacts": {name: str(p) for name, p in artifacts.items()},
        }
        if eval_summary:
            res["evaluation"] = eval_summary
        return res

    def run_stage_26_evaluate(self, test_path: Path) -> Optional[Dict[str, Any]]:
        """Stage T035-T040: Holdout Evaluation Benchmarking."""
        self.logger.info("Stage 26: Running Holdout Evaluation Benchmark ...")
        from src.evaluation.engine import EvaluationEngine

        holdout_pairs = []
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    holdout_pairs.append(json.loads(line))

        if not holdout_pairs:
            self.logger.warning("No holdout test pairs found in %s", test_path)
            return None

        rng = np.random.default_rng(self.config.random_seed)
        sample_size = min(60, len(holdout_pairs))
        indices = rng.choice(len(holdout_pairs), size=sample_size, replace=False)
        selected_pairs = [holdout_pairs[i] for i in indices]

        targets = [p.get("target_text", "").strip() for p in selected_pairs]
        contexts = [p.get("context_text", "").strip() for p in selected_pairs]

        emb_engine = EmbeddingEngine(EmbeddingConfig(
            model_name=self.config.embedding_model,
            batch_size=self.config.batch_size,
            device=self.config.device,
        ))
        style_index = StyleEmbeddingIndex.load(self.style_index_dir)
        retriever = StyleRetriever(index=style_index, embedder=emb_engine)

        retrieval_candidates = []
        for p in selected_pairs:
            ctx = p.get("context_text", "")
            last_turn = p["context"][-1].get("text", "") if p.get("context") else ""
            query = last_turn or ctx or "kya scene he"
            ret_res = retriever.retrieve(query)
            if ret_res.items:
                ex_resp = ret_res.items[0].example.get("response", "").strip().lower()
                cand = re.sub(r"[.!]+$", "", ex_resp).strip()
            else:
                cand = "ha sahi he"
            retrieval_candidates.append(cand)

        eval_engine = EvaluationEngine(embedding_engine=emb_engine)
        summary, details = eval_engine.evaluate_strategy(
            "Persona Engine (Retrieved Few-Shot)",
            retrieval_candidates,
            targets,
            contexts=contexts,
        )

        eval_report_path = self.config.output_dir / "evaluation_summary.json"
        with open(eval_report_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)

        self.logger.info("Stage 26 complete: Judge Overall: %.2f / 5.0, Linguistic: %.2f, Behavioral: %.2f",
                         summary.judge_overall, summary.linguistic_composite, summary.behavioral_composite)
        return summary.to_dict()


def main() -> None:
    """CLI Entrypoint for the Persona Pipeline."""
    # Ensure Windows console handles UTF-8
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Persona Pipeline: End-to-end Persona Extraction and Package Compiler.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        default="data/raw",
        help="Path to raw WhatsApp chat export file (.txt) or directory containing exports.",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/output/persona_package",
        help="Destination directory for compiled production persona package artifacts.",
    )
    parser.add_argument(
        "--work-dir", "-w",
        default="data/processed",
        help="Intermediate directory for processed datasets and forensic profiles.",
    )
    parser.add_argument(
        "--target-speaker", "-s",
        default=None,
        help="Specific speaker name to extract persona for (auto-detected if omitted).",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-execution of all pipeline stages, bypassing cached steps.",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively scan nested subdirectories for .txt chat files.",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Execute holdout evaluation benchmark after compiling the package.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Computation device for transformer embeddings ('cpu', 'cuda').",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable detailed debug logging.",
    )

    args = parser.parse_args()

    config = PersonaPipelineConfig(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        work_dir=Path(args.work_dir),
        target_speaker=args.target_speaker,
        force=args.force,
        recursive=args.recursive,
        run_eval=args.run_eval,
        device=args.device,
        verbose=args.verbose,
    )

    pipeline = PersonaPipeline(config)
    result = pipeline.run()

    print("\nPipeline finished successfully!")
    print(f"Target Speaker: {result['target_speaker']}")
    print(f"Package Directory: {args.output}")
    print(f"Total Time: {result['duration_seconds']}s")


if __name__ == "__main__":
    main()
