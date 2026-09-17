# Persona Extraction & Profiling Pipeline

An automated, forensic NLP and machine learning pipeline that extracts an authentic communication persona, behavioral model, and production system prompt from raw WhatsApp conversation exports (`.txt`).

---

## 1. Quick Start

### Step 1: Place Raw WhatsApp Export Files
Drop your exported WhatsApp `.txt` chat files into `data/raw/` (or keep them in any folder of your choice):
```text
data/raw/
├── friend_chat.txt
├── project_group_chat.txt
└── family_chat.txt
```

### Step 2: Run the Unified Pipeline
Run the pipeline with default settings:
```bash
python src/pipeline.py
```
That is it! The pipeline automatically:
1. Ingests and parses all WhatsApp text files.
2. Canonicalizes messages and builds sender statistics.
3. Auto-detects the target speaker across chats (defaulting to `You` or the primary chat participant).
4. Sanitizes sensitive PII (URLs, emails, phone numbers, credentials).
5. Cleans and normalizes text while preserving casing, emojis, and slang.
6. Segments conversation sessions by temporal gaps.
7. Reconstructs context-response pairs.
8. Filters low-quality noise and performs stratified sampling.
9. Partitions into Train (70%), Dev (15%), and Holdout (15%) splits with zero session leakage.
10. Generates 768-dimensional transformer embeddings.
11. Performs Spherical K-Means clustering, topic discovery, surface linguistic profiling, syntax modeling, Hinglish code-switching analysis, discourse tracking, situational classification, behavioral extraction, and style profiling.
12. Synthesizes structured epistemic inferences (`OBSERVED`, `INFERRED`, `UNKNOWN`).
13. Selects representative exemplars and builds a 768-dim few-shot retrieval index.
14. Compiles the production persona package into `data/output/persona_package/`.

---

## 2. CLI Options & Customization

The CLI supports custom input locations, output targets, explicit speaker overrides, caching controls, and evaluation benchmarks:

```bash
# Run on a specific raw file
python src/pipeline.py --input "data/raw/my_chat.txt"

# Run on a custom input folder and write to a custom output directory
python src/pipeline.py --input "path/to/chats" --output "dist/my_persona"

# Explicitly specify target speaker (disabling auto-detection)
python src/pipeline.py --target-speaker "Aditya Kathe"

# Force re-running all stages from scratch (bypassing cached steps)
python src/pipeline.py --force

# Run with holdout evaluation benchmark scoring
python src/pipeline.py --run-eval

# Recursively scan subdirectories for .txt files
python src/pipeline.py --recursive
```

### Complete CLI Argument Reference

| Flag | Shorthand | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | `data/raw` | Path to raw WhatsApp export file (`.txt`) or directory. |
| `--output` | `-o` | `data/output/persona_package` | Destination directory for compiled persona artifacts. |
| `--work-dir` | `-w` | `data/processed` | Working directory for intermediate processed datasets. |
| `--target-speaker` | `-s` | `None` (auto-detect) | Target speaker name to profile. |
| `--force` | `-f` | `False` | Force re-execution of all stages, bypassing cache. |
| `--recursive` | `-r` | `False` | Search nested directories for `.txt` files. |
| `--run-eval` | | `False` | Run holdout evaluation benchmarking on test split. |
| `--device` | | `cpu` | PyTorch device (`cpu` or `cuda`). |
| `--verbose` | `-v` | `False` | Enable verbose debug logging. |

---

## 3. Programmatic Python API

You can also import and run the pipeline inside any Python application or automated service:

```python
from pathlib import Path
from src.pipeline import PersonaPipeline, PersonaPipelineConfig

config = PersonaPipelineConfig(
    input_path=Path("data/raw"),
    output_dir=Path("data/output/persona_package"),
    target_speaker=None,  # Auto-detected if None
    force=False,          # Uses caching when available
    run_eval=False,
)

pipeline = PersonaPipeline(config)
result = pipeline.run()

print(f"Target Speaker: {result['target_speaker']}")
print(f"Duration: {result['duration_seconds']}s")
print(f"Compiled Artifacts: {result['artifacts']}")
```

---

## 4. Production Output Artifacts

All production artifacts are compiled into `data/output/persona_package/`:

| Artifact | Type | Description |
| :--- | :--- | :--- |
| `style_profile.json` | JSON | Quantitative style rules (casing, terminal punctuation drop rate, multi-bubble burstiness, code-switching ratio, emoji frequencies). |
| `linguistic_stats.json` | JSON | Forensic surface linguistic and syntactic distributions (speech acts, clause structures, negations, pronoun ratios). |
| `behavior_profile.json` | JSON | 10 quantified behavioral dimensions (directness, verbosity, hedging, confidence, disagreement, humor, tech depth, formality) with epistemic categories. |
| `vocabulary.json` | JSON | Authentic lexical inventory (discourse connectives, slang, characteristic collocations, technical nouns, top emojis). |
| `system_prompt.md` | Markdown | Production-ready LLM system prompt encoding voice constraints, tone, situational modulations, and few-shot slots. |
| `persona_report.md` | Markdown | Comprehensive human-readable forensic synthesis report for review. |
| `package_metadata.json` | JSON | Compilation timestamps, artifact SHA-256 checksums, and provenance tracking. |

---

## 5. Privacy & Security

The pipeline includes deterministic PII masking (`src/privacy/pii_filter.py`) and post-compilation verification:
- Automatic masking of URLs, emails, phone numbers, UPI handles, API keys, OTPs, card numbers, and passwords.
- Public persona artifacts are strictly verified to contain zero sensitive credentials or unmasked entities.

---

## 6. Running Tests

Run the full automated test suite (27 test modules covering all pipeline components):

```bash
pytest tests/ -q
```
