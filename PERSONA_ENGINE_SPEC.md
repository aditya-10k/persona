# Persona Extraction Engine
## Technical Specification & Task-by-Task Implementation Plan

**Document status:** Source of Truth  
**Project:** Standalone WhatsApp → Communication Persona Extraction Engine  
**Primary language:** Python  
**Target hardware:** Consumer laptop with integrated GPU; CPU-first, GPU-optional  
**Downstream consumer:** AI Spokesperson / Portfolio Agent  
**Current phase:** T001–T017 implemented and validated; next task is T018 Code-Switching / Language Analysis

---

# 0. PROJECT CHARTER

## 0.1 Objective

Build a standalone, reusable NLP/ML system that takes a user's WhatsApp chat exports and produces a structured, evidence-backed representation of how that user communicates.

The system is intended to model **communication style and conversational behavior**, not to reproduce or expose the user's private life.

The final output will later be consumed by a separate AI spokesperson system so that the spokesperson can combine:

1. **Professional knowledge** — resume, projects, experience, education, achievements, links, etc.
2. **Communication persona** — extracted from WhatsApp.
3. **Runtime context** — the current conversation and requested action.

The Persona Extraction Engine itself must remain independent of the spokesperson application.

---

## 0.2 Problem Definition

A naive "WhatsApp → LLM → personality description" approach is insufficient.

The system must distinguish between:

- what the user talks about,
- what information the user knows,
- and how the user communicates.

The target is therefore a **computational communication profile** supported by measurable linguistic evidence, contextual examples, semantic representations, and higher-level behavioral inference.

---

## 0.3 Expected Output

The engine must ultimately produce a versioned Persona Package containing:

```text
persona_package/
├── style_profile.json
├── behavior_profile.json
├── linguistic_stats.json
├── vocabulary.json
├── situational_styles.json
├── style_examples.json
├── embeddings/
├── evaluation_report.json
└── persona_report.md
```

The exact schema may evolve during implementation, but changes must be recorded in the Decision Log.

---

## 0.4 Non-Goals

The system must NOT:

- infer sensitive personal attributes unnecessarily;
- attempt psychological diagnosis;
- expose private conversation content to the downstream public agent;
- memorize private facts merely because they occur in WhatsApp;
- treat sentiment as personality;
- treat word frequency as sufficient evidence of style;
- rely exclusively on an LLM's subjective description;
- require a GPU;
- perform expensive LLM/RAG operations for every runtime interaction;
- couple WhatsApp processing directly to the final portfolio application.

---

## 0.5 Design Principles

### Principle 1 — Evidence over intuition

Every important persona claim should ideally have measurable evidence or representative examples.

### Principle 2 — Context matters

Communication style is situational. Casual, technical, professional, argumentative, humorous, explanatory, etc. may have different characteristics.

### Principle 3 — Modern NLP first

Transformer-based contextual representations, embeddings, semantic methods, modern classifiers, and LLM-assisted structured inference are preferred for semantic/high-level analysis.

Classical NLP remains valid for measurable linguistic signals such as punctuation, vocabulary statistics, sentence structure, frequency distributions, and parsing.

### Principle 4 — Privacy by design

Raw WhatsApp data remains local unless explicitly and deliberately passed to an external model after appropriate sanitization/sampling.

### Principle 5 — Batch expensive computation

Embeddings and analysis should be generated offline in batches where possible.

### Principle 6 — Reproducibility

Every analysis should be reproducible from a known dataset version, configuration, model, and processing version.

### Principle 7 — Evaluation is part of the system

A persona extractor without evaluation cannot establish that the extracted persona is accurate.

---

# 1. SYSTEM OVERVIEW

## 1.1 Complete Architecture

```text
WhatsApp Export (.txt)
        |
        v
+-----------------------+
| Extraction            |
| Parser                |
+----------+------------+
           |
           v
+-----------------------+
| Transformation        |
| Canonical schema      |
+----------+------------+
           |
           v
+-----------------------+
| Validation             |
+----------+------------+
           |
           v
+-----------------------+
| Privacy / Cleaning    |
+----------+------------+
           |
           v
+-----------------------+
| Conversation          |
| Reconstruction        |
+----------+------------+
           |
           v
+-----------------------+
| Dataset Construction  |
+----------+------------+
           |
           +----------------------+
           |                      |
           v                      v
+----------------------+  +----------------------+
| Linguistic Analysis  |  | Transformer /        |
| Statistical Signals  |  | Embedding Analysis   |
+----------+-----------+  +----------+-----------+
           |                         |
           +------------+------------+
                        |
                        v
              +----------------------+
              | Behavioral /         |
              | Situational Analysis |
              +----------+-----------+
                         |
                         v
              +----------------------+
              | Persona Inference    |
              | + LLM Structured     |
              |   Analysis           |
              +----------+-----------+
                         |
                         v
              +----------------------+
              | Persona Package      |
              +----------+-----------+
                         |
                         v
              +----------------------+
              | Evaluation           |
              | + Holdout Testing    |
              +----------------------+
```

---

# 2. HARDWARE AND COMPUTE STRATEGY

The development machine has an integrated GPU rather than a discrete NVIDIA GPU.

Therefore:

## Local by default

- parsing;
- transformation;
- validation;
- cleaning;
- privacy filtering;
- conversation segmentation;
- statistical NLP;
- lightweight linguistic parsing;
- dataset generation;
- clustering where computationally practical;
- evaluation;
- artifact generation.

## API / remote compute where beneficial

- large transformer inference;
- expensive LLM reasoning;
- large-scale semantic analysis;
- potentially large embedding jobs if local CPU performance is insufficient.

## Local transformer models

Small/efficient transformer models may be tested locally using CPU or available acceleration, but the architecture must never depend on local GPU availability.

The project must support a configurable model/provider layer.

---

# 3. DATA ACQUISITION

## T001 — Acquire WhatsApp Dataset

### Goal

Obtain an authentic WhatsApp exported `.txt` dataset without media.

### Input

WhatsApp chat export.

### Requirements

- Preserve original raw file unchanged.
- Store it only under `data/raw/`.
- Never commit raw chat data to Git.
- Record metadata without storing sensitive content.

### Output

```text
data/raw/whatsapp.txt
```

### Acceptance criteria

- Raw export exists.
- File can be opened and parsed.
- Original file is preserved.
- `.gitignore` prevents accidental commit.

---

# 4. EXTRACTION & TRANSFORMATION

## T002 — Build Robust WhatsApp Parser

### Status

**COMPLETE — implemented and tested.**

### Goal

Convert raw WhatsApp export text into structured message records.

### Implementation

Implemented at:

```text
src/parser/whatsapp_parser.py
src/parser/__init__.py
tests/test_whatsapp_parser.py
```

The parser supports:

- common WhatsApp date formats;
- 12-hour and 24-hour timestamps;
- sender extraction;
- multiline messages;
- group conversations;
- system-generated messages;
- media placeholders;
- deleted messages;
- call records;
- forwarded messages;
- Unicode and emojis;
- punctuation;
- messages containing commas/colons;
- malformed/orphan lines;
- URLs and technical/code-like message content;
- deterministic directory processing.

### Important message-boundary rule

Message boundaries are determined by valid WhatsApp message headers.

A non-header physical line continues the previous message.

`[Forwarded]` does **not** create a special multiline block. If each physical line has its own valid timestamp/header, each is a separate message. If subsequent lines do not have a valid header, they remain part of the current message.

### System-event handling

WhatsApp system-generated records may not contain a sender.

Examples include:

```text
- [Call]
- [System notification]
- User A left
- User B created the group
- You added User A
- You changed the group name to '...'
```

The parser preserves these records and their provenance rather than dropping them.

The parser can emit:

```text
message header has no sender delimiter
```

for system-generated records whose WhatsApp header has no sender field. T004 now classifies these as expected system-event warnings when the record is independently recognizable as a system event.

### Implemented record metadata

The implementation preserves:

```json
{
  "message_id": 0,
  "timestamp": "ISO-8601",
  "sender": "string or null",
  "text": "string",
  "message_type": "text",
  "is_system": false,
  "is_media": false,
  "is_forwarded": false,
  "raw_line_start": 1,
  "raw_line_end": 1,
  "source_file": "string",
  "parse_warnings": []
}
```

### Validation performed

The parser test suite passes:

```text
13 tests passed
```

The parser was also run across the full raw dataset.

Per-file extracted record counts:

```text
chat_01.txt                                  1,676
chat_02.txt                               1,099
chat_03.txt                          2,457
chat_04.txt  2,366
chat_05.txt                                   719
chat_06.txt                                      486
chat_07.txt                           15,936
chat_08.txt                        7,390
chat_09.txt                                264
chat_10.txt                                   127
chat_11.txt                       34,133
chat_12.txt                     2,472
chat_13.txt                                    4,735
```

### Acceptance status

- All parseable messages extracted.
- Multiline boundaries verified.
- Sender/timestamp boundaries verified.
- Parser warnings preserved.
- Deterministic output implemented.
- Representative tests pass.
- Raw source data remains unchanged.

---

## T003 — Build Transformation Layer

### Status

**COMPLETE — implemented and validated.**

### Goal

Convert parser output into a stable canonical dataset.

### Implementation

Implemented at:

```text
src/transformation/canonicalize.py
```

The transformation layer performs:

- timestamp normalization;
- message type normalization;
- NFC Unicode normalization;
- whitespace handling only where semantically safe;
- stable canonical IDs;
- source file tracking;
- raw line provenance;
- sender statistics;
- preservation of parser warnings.

Canonical IDs use a deterministic source-local format:

```text
<source_file>:<local_index>
```

Example:

```text
chat_01.txt:00000001
```

### Outputs

```text
data/processed/messages.jsonl
data/processed/sender_stats.json
```

### Important preservation rule

`parse_warnings` are retained in the canonical dataset so downstream validation can distinguish genuine parser problems from expected WhatsApp export artifacts.

The transformation layer does not perform privacy filtering, persona inference, or destructive style normalization.

### Acceptance status

- Stable IDs implemented.
- Timestamp format consistent.
- Original message text preserved.
- Parser warnings preserved.
- Source line provenance preserved.
- Sender statistics generated.
- Canonical JSONL successfully generated for the full dataset.

---

# 5. VALIDATION

## T004 — Build Dataset Validator

### Status

**COMPLETE — implemented and validated against the full dataset.**

### Goal

Detect parser/transformation errors before NLP is performed.

### Implementation

Implemented at:

```text
src/validation/validator.py
```

The validator checks:

- malformed JSON records;
- required fields;
- parser warnings;
- expected system-event parser warnings;
- timestamps;
- timestamp monotonicity per source/chat;
- missing senders;
- empty messages;
- message types;
- exact duplicates;
- potential duplicates;
- suspiciously large messages;
- source/line provenance;
- dataset statistics.

### Expected parser-warning handling

The validator distinguishes between:

```text
genuine parser errors
```

and:

```text
expected WhatsApp system-event warnings
```

The warning:

```text
message header has no sender delimiter
```

is considered expected only when:

1. the warning is exactly the known warning;
2. the record has no sender;
3. the message text is independently classified as a recognized WhatsApp system event.

This prevents ordinary messages containing words such as `added`, `removed`, or `left` from being incorrectly classified as system events.

### Conservative system-event classification

T004 recognizes the verified system-event categories:

```text
call_event
system_notification
group_management_event
encryption_notice
```

The classifier is intentionally conservative.

In particular, group-management events require the expected WhatsApp `- ` system-event structure. Keyword occurrence alone is not sufficient.

An initial broader classifier over-classified 295 records. The classifier was tightened and the verified result is now 184 system events.

### Full-dataset validation result

The current canonical dataset contains:

```text
67,209 total messages
```

Current validation result:

```json
{
  "status": "PASS",
  "total_messages": 67209,
  "parse_error_count": 0,
  "total_parse_warning_count": 106,
  "expected_system_event_warning_count": 106,
  "malformed_record_count": 0,
  "schema_error_count": 0,
  "missing_timestamp_count": 0,
  "missing_sender_count": 0,
  "system_event_count": 184,
  "expected_system_preamble_count": 10,
  "empty_message_count": 0,
  "exact_duplicate_group_count": 0,
  "potential_duplicate_group_count": 81,
  "suspicious_large_message_count": 1,
  "unexpected_message_type_count": 0,
  "timestamp_anomaly_count": 0,
  "provenance_error_count": 0
}
```

System-event breakdown:

```text
78  call events
35  system notifications
61  group-management events
10  encryption notices
---
184 senderless system events
```

Parser-warning breakdown:

```text
106 expected system-event warnings
0   genuine parser errors
```

### Informational findings

The following do not cause validation failure under the current validator design:

- 81 potential duplicate groups;
- 1 suspiciously large message.

Potential duplicates can be legitimate repeated messages. Large messages are flagged for inspection but are not automatically invalid.

### Outputs

```text
data/processed/validation_report.json
```

### Acceptance status

- Full dataset validates successfully.
- No malformed records.
- No schema errors.
- No missing timestamps.
- No genuine missing senders.
- No genuine parser errors.
- No unexpected message types.
- No timestamp ordering anomalies.
- No provenance errors.
- System-event false positives were corrected from 295 to the verified 184.
- T002 and T003 outputs were left unchanged while the T004 classification logic was corrected.

# 6. PRIVACY AND CLEANING

## T005 — Privacy Filter

### Goal

Prevent private/sensitive information from becoming part of the public persona artifact.

### Detect/remove or mask where appropriate

- phone numbers;
- email addresses;
- URLs when not useful to style analysis;
- addresses;
- OTPs;
- passwords;
- API keys;
- authentication tokens;
- financial identifiers;
- highly private third-party information.

### Critical distinction

Some tokens that look sensitive may also be useful stylistically.

The system should therefore preserve:

- slang;
- abbreviations;
- emojis;
- intentional spelling;
- punctuation;
- casing;
- Hinglish/code-switching;
- profanity when relevant to style.

### Output

```text
data/processed/sanitized_messages.jsonl
```

The raw dataset remains separate and local.

---

## T006 — Text Cleaning / Normalization

### Goal

Create analysis-specific representations without destroying the original.

Maintain multiple views:

```text
raw_text
normalized_text
tokenized_text
analysis_text
```

Never overwrite raw text.

### Example

```text
raw:
"broooo this is FUCKED 😭😭"

normalized:
"broooo this is FUCKED 😭😭"

analysis:
tokenized representation preserving lexical/style metadata
```

Do NOT automatically lowercase everything because casing is itself a style signal.

---

# 7. CONVERSATION RECONSTRUCTION

## T007 — Temporal Conversation Segmentation

### Goal

Group messages into meaningful conversational sessions.

### Signals

Primary:

- temporal gaps.

Secondary:

- participant changes;
- conversational continuity;
- semantic continuity where useful.

The time-gap threshold must be configurable rather than hardcoded.

### Output

```text
data/processed/conversations.jsonl
```

---

## T008 — Context Reconstruction

### Goal

Create multi-turn context-response examples.

Example:

```json
{
  "conversation_id": "c001",
  "turns": [
    {
      "speaker": "other",
      "text": "..."
    },
    {
      "speaker": "user",
      "text": "..."
    }
  ]
}
```

Generate training/analysis units such as:

```json
{
  "context": ["message A", "message B", "message C"],
  "target_response": "user response",
  "metadata": {}
}
```

Context windows should be configurable.

Do not reduce every interaction to a single previous message.

---

# 8. DATASET CONSTRUCTION

## T009 — Quality Filtering

Filter or flag:

- media-only messages;
- extremely low-information messages;
- system messages;
- duplicates;
- spam;
- accidental exports;
- corrupted records.

Do not blindly remove all short messages.

Short responses may contain important style information.

---

## T010 — Sampling Strategy

The analysis dataset should avoid overrepresenting the most common conversational context.

Sampling should consider:

- conversation;
- time period;
- conversation type;
- message length;
- semantic cluster;
- situational category.

This reduces the risk of extracting a persona dominated by one repetitive chat.

---

## T011 — Train / Development / Holdout Split

Where model-based classification or generation is evaluated:

```text
Training / analysis set
Development set
Held-out evaluation set
```

Splitting should preferably occur at the **conversation level**, not randomly by individual message, to reduce context leakage.

---

# 9. MODERN NLP ANALYSIS

## T012 — Transformer Embedding Layer

### Goal

Generate contextual semantic representations of messages and conversations.

Requirements:

- model must be configurable;
- embeddings must be versioned;
- batch generation;
- caching;
- deterministic metadata;
- no need for GPU.

Store:

```text
embedding_model
model_version
embedding_dimension
source_id
vector
```

Potential model selection must be evaluated rather than assumed.

---

## T013 — Semantic Similarity

Measure similarity between:

- messages;
- responses;
- conversations;
- style examples;
- generated responses.

Use transformer embeddings rather than relying primarily on TF-IDF.

---

## T014 — Semantic Clustering

Cluster conversational material in embedding space.

Potential approaches:

- density-based clustering;
- hierarchical clustering;
- dimensionality reduction for inspection.

The algorithm should be chosen based on dataset behavior rather than popularity.

Goal:

Discover naturally occurring conversational contexts rather than forcing arbitrary categories.

---

## T015 — Topic Discovery

Use modern embedding-based topic discovery where useful.

Possible approach:

- transformer embeddings;
- clustering;
- BERTopic-style topic modeling.

Topics describe **what is being discussed**, not communication style.

This distinction must remain explicit.

---

# 10. LINGUISTIC FINGERPRINTING

## T016 — Surface Linguistic Profile

Measure:

- character counts;
- word counts;
- sentence lengths;
- message lengths;
- vocabulary richness;
- lexical diversity;
- punctuation;
- capitalization;
- contractions;
- abbreviations;
- emoji patterns;
- repeated expressions;
- spelling variants.

These features remain valuable as quantitative evidence.

---

## T017 — Syntactic Profile

Analyze:

- POS distributions;
- dependency structures;
- sentence complexity;
- clause structure;
- fragments;
- coordination;
- questions;
- imperatives;
- negation;
- conditional constructions.

Goal:

Determine how the user constructs language, not just which words they use.

---

## T018 — Code-Switching / Language Analysis

Measure:

- English usage;
- Hindi usage;
- Hinglish;
- other languages if present;
- code-switch frequency;
- code-switch position;
- vocabulary borrowed across languages.

The detector must be tolerant of informal spellings.

---

## T019 — Discourse Analysis

Analyze conversational functions such as:

- answering;
- questioning;
- explaining;
- clarifying;
- correcting;
- disagreeing;
- agreeing;
- joking;
- hedging;
- giving advice;
- expressing uncertainty;
- changing topic;
- emphasizing.

This is a major component of the persona model.

---

# 11. BEHAVIORAL AND SITUATIONAL ANALYSIS

## T020 — Situational Classification

Identify contexts such as:

```text
casual
technical
professional
explanatory
argumentative
humorous
advice
frustration
excitement
uncertainty
greeting
acknowledgement
```

The final categories may be discovered/expanded based on the data.

Use transformer/LLM-assisted classification where appropriate.

---

## T021 — Behavioral Pattern Extraction

Infer patterns such as:

- directness;
- verbosity;
- hedging;
- explanation structure;
- disagreement style;
- humor usage;
- example usage;
- question frequency;
- confidence expression;
- emotional expressiveness;
- conversational initiative.

Each inferred behavior should retain:

- score;
- confidence;
- supporting examples;
- analysis method.

---

# 12. PERSONA INFERENCE

## T022 — Build Global Style Profile

Create measurable dimensions such as:

```json
{
  "directness": 0.0,
  "formality": 0.0,
  "verbosity": 0.0,
  "humor": 0.0,
  "sarcasm": 0.0,
  "hedging": 0.0,
  "emotional_expression": 0.0
}
```

Values must be derived from the analysis rather than manually invented.

---

## T023 — Build Situational Style Profiles

Instead of assuming one universal style:

```text
global
├── casual
├── technical
├── professional
├── explanatory
├── argumentative
└── humorous
```

Each profile contains its own characteristics and evidence.

---

## T024 — LLM-Assisted Structured Inference

Use an LLM for higher-level interpretation that is difficult to derive purely statistically.

Input should consist of selected/sanitized examples and quantitative features.

Require structured JSON output.

The model should NOT be allowed to invent unsupported facts.

Prompt should explicitly distinguish:

```text
observed
inferred
unknown
```

---

## T025 — Evidence Aggregation

Combine:

- statistical linguistic features;
- transformer representations;
- classifier outputs;
- representative examples;
- LLM analysis.

Create confidence scores.

Example:

```json
{
  "trait": "directness",
  "score": 0.86,
  "confidence": 0.91,
  "evidence_count": 438,
  "supporting_examples": ["..."]
}
```

---

# 13. STYLE EXAMPLES

## T026 — Build Representative Style Dataset

Each example should contain:

```json
{
  "example_id": "s001",
  "category": "technical",
  "context": "...",
  "response": "...",
  "style_tags": [],
  "embedding_model": "...",
  "source_conversation": "..."
}
```

Examples must be:

- representative;
- privacy-filtered;
- diverse;
- non-duplicative;
- context-aware.

---

## T027 — Example Selection

Do not simply select the most common messages.

Select examples covering:

- different situations;
- different sentence lengths;
- different conversational functions;
- distinctive but representative style;
- technical/professional contexts where available.

---

# 14. EMBEDDING AND RETRIEVAL

## T028 — Style Embedding Index

Create an embedding index for style examples.

Possible implementation:

- PostgreSQL + pgvector;
- local vector index;
- another vector store if justified.

The exact implementation is a decision to be finalized based on downstream integration.

---

## T029 — Style Retrieval

Given a new context:

```text
user prompt
    ↓
context representation
    ↓
retrieve relevant style examples
    ↓
generation
```

Retrieval should be selective.

Do NOT perform expensive style retrieval on every simple request if the static persona profile is sufficient.

---

# 15. PERSONA PACKAGE

## T030 — style_profile.json

Contains global and situational communication traits.

---

## T031 — linguistic_stats.json

Contains quantitative measurements and distributions.

---

## T032 — behavior_profile.json

Contains behavioral/discourse traits with confidence/evidence.

---

## T033 — vocabulary.json

Contains:

- common phrases;
- abbreviations;
- stylistically distinctive expressions;
- slang;
- language-mixing patterns.

Private factual content must not be retained merely because it is frequent.

---

## T034 — persona_report.md

Human-readable summary containing:

- methodology;
- dataset statistics;
- global style;
- situational style;
- linguistic fingerprint;
- discourse behavior;
- limitations;
- confidence;
- representative examples.

---

# 16. EVALUATION

Evaluation is mandatory.

## T035 — Holdout Dataset

Reserve conversations not used during persona inference.

---

## T036 — Linguistic Similarity Evaluation

Compare generated persona responses with held-out real responses using:

- sentence-length distributions;
- punctuation patterns;
- lexical distributions;
- syntactic characteristics;
- embedding similarity.

No single metric determines success.

---

## T037 — Behavioral Evaluation

Evaluate whether generated outputs reproduce expected behaviors:

- directness;
- verbosity;
- humor;
- hedging;
- explanation structure;
- disagreement style;
- situational behavior.

---

## T038 — LLM-as-Judge

Use structured evaluation prompts to compare:

```text
real response
generated response
```

against a predefined rubric.

The judge must not be the only evaluation mechanism.

---

## T039 — Human Evaluation

Where possible, evaluate:

- Does it sound like the user?
- Is the response natural?
- Is the situation appropriate?
- Is the style recognizable?
- Does it preserve the user's communication habits without copying private facts?

---

## T040 — Failure Analysis

Every major evaluation run should identify failure categories.

Examples:

```text
too formal
too verbose
too generic
wrong humor
wrong slang
incorrect context
wrong language mix
overuse of distinctive phrases
semantic mismatch
```

---

# 17. PERFORMANCE

## T041 — Batch Processing

All expensive offline processing should support batching.

---

## T042 — Caching

Cache:

- parsed messages;
- cleaned messages;
- embeddings;
- classifier outputs;
- LLM analyses.

Changing one downstream stage should not require reprocessing everything.

---

## T043 — Incremental Processing

The architecture should eventually allow a new WhatsApp export to be processed without rebuilding unchanged artifacts unnecessarily.

---

# 18. REPOSITORY STRUCTURE

```text
persona-engine/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── output/
│
├── src/
│   ├── parser/
│   │   ├── __init__.py
│   │   └── whatsapp_parser.py
│   │
│   ├── transformation/
│   │   ├── __init__.py
│   │   └── canonicalize.py
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   └── validator.py
│   │
│   ├── privacy/
│   │   ├── __init__.py
│   │   └── pii_filter.py
│   │
│   ├── segmentation/
│   │   ├── __init__.py
│   │   └── conversations.py
│   │
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── sampler.py
│   │   └── examples.py
│   │
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── linguistic.py
│   │   ├── embeddings.py
│   │   ├── semantic.py
│   │   ├── topics.py
│   │   ├── language.py
│   │   └── discourse.py
│   │
│   ├── persona/
│   │   ├── __init__.py
│   │   ├── behavior.py
│   │   ├── style.py
│   │   ├── inference.py
│   │   └── examples.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── judge.py
│   │   └── failure_analysis.py
│   │
│   └── pipeline.py
│
├── tests/
│
├── configs/
│   └── default.yaml
│
├── README.md
├── requirements.txt
└── PERSONA_ENGINE_SPEC.md
```

---

# 19. IMPLEMENTATION ORDER

The final system is designed as one complete system, but implementation proceeds in dependency order.

```text
T001 Environment + Dataset
  ↓
T002 Parser [COMPLETE]
  ↓
T003 Transformation [COMPLETE]
  ↓
T004 Validation [COMPLETE]
  ↓
T005 Privacy
  ↓
T006 Cleaning
  ↓
T007 Segmentation
  ↓
T008 Context Reconstruction
  ↓
T009 Quality Filtering
  ↓
T010 Sampling
  ↓
T011 Dataset Split
  ↓
T012 Transformer Embeddings
  ↓
T013 Semantic Similarity
  ↓
T014 Clustering
  ↓
T015 Topic Discovery
  ↓
T016 Linguistic Fingerprint
  ↓
T017 Syntactic Analysis
  ↓
T018 Language / Code Switching
  ↓
T019 Discourse Analysis
  ↓
T020 Situational Classification
  ↓
T021 Behavioral Analysis
  ↓
T022 Global Style
  ↓
T023 Situational Style
  ↓
T024 LLM Inference
  ↓
T025 Evidence Aggregation
  ↓
T026-T027 Style Examples
  ↓
T028-T029 Retrieval
  ↓
T030-T034 Persona Package
  ↓
T035-T040 Evaluation
  ↓
T041-T043 Performance / Incremental Processing
```

---

# 20. DEVELOPMENT RULES FOR FUTURE SESSIONS

These rules are binding for work on this project.

## Rule 1 — Do not silently change architecture

If an implementation reveals that an architectural decision is wrong, explicitly flag it and update the Decision Log before changing it.

## Rule 2 — Do not downgrade the NLP scope

Do not replace modern contextual/transformer methods with simplistic keyword/rule-based approaches merely because they are easier.

Classical techniques can remain as supporting measurements.

## Rule 3 — Do not add random ML techniques

Every model must have a defined purpose and evaluation method.

## Rule 4 — Do not send raw WhatsApp data externally by default

External inference requires deliberate sanitization/sampling.

## Rule 5 — Preserve raw data

Never overwrite the original WhatsApp export.

## Rule 6 — Preserve provenance

Every derived artifact should be traceable to:

- dataset version;
- processing version;
- model;
- configuration.

## Rule 7 — No "magic persona"

The final persona must be explainable through evidence.

## Rule 8 — Contextual style is mandatory

Do not collapse the entire user's communication style into one generic tone score.

## Rule 9 — Evaluation cannot be skipped

The system must demonstrate that the extracted persona is useful.

## Rule 10 — Do not prematurely integrate with the spokesperson

The Persona Engine must first produce a stable, tested output contract.

---

# 21. DECISION LOG

## D001 — Standalone Persona Engine

**Decision:** WhatsApp analysis is a separate project/subsystem.

**Reason:** It is independently useful, easier to evaluate, reusable, and keeps private-data processing separate from the portfolio application.

---

## D002 — Python

**Decision:** Python is the primary implementation language.

**Reason:** Strong NLP/ML ecosystem and suitability for offline data processing.

---

## D003 — CPU-first architecture

**Decision:** The engine must function without an NVIDIA GPU.

**Reason:** Development machine has integrated GPU only.

---

## D004 — Modern NLP

**Decision:** Transformer-based representations and modern semantic methods form the core of high-level NLP analysis.

**Reason:** The goal is an accurate modern communication model rather than a basic traditional NLP demonstration.

---

## D005 — Classical NLP remains supporting infrastructure

**Decision:** Regex, frequency statistics, POS/dependency analysis, punctuation statistics, and similar methods remain in the pipeline where they provide measurable information.

**Reason:** These signals are still useful for linguistic fingerprinting and validation.

---

## D006 — WhatsApp is style data, not knowledge data

**Decision:** WhatsApp-derived artifacts primarily represent communication behavior/style.

**Reason:** The future spokesperson's factual knowledge should come from professional sources.

---

## D007 — Evaluation with held-out conversations

**Decision:** Persona inference must reserve evaluation data.

**Reason:** Prevents circular validation and gives a better estimate of generalization.

---

## D008 — Preserve and classify WhatsApp system events

**Decision:** WhatsApp system-generated records without senders remain in the canonical dataset and are classified during validation rather than discarded.

**Reason:** These records are valid export data and are useful for provenance and message-boundary validation. They should not be mistaken for authored conversation messages.

---

## D009 — Conservative system-event detection

**Decision:** T004 uses deterministic structural signals to identify expected WhatsApp system events. It does not classify a message as a system event merely because normal conversation text contains words such as `added`, `removed`, or `left`.

**Reason:** Validation should minimize false positives. The observed WhatsApp export format provides stronger structural evidence than keyword occurrence.

---

## D010 — Preserve parser warnings through T003

**Decision:** `parse_warnings` are retained in `messages.jsonl`.

**Reason:** T004 needs the original parser diagnostics to distinguish expected WhatsApp export artifacts from genuine parser errors without modifying the canonical dataset.

---

## D011 — T002/T003 outputs are not changed to fix T004 classification

**Decision:** The T004 system-event classification issue was fixed in the validator rather than altering the parser or canonical dataset.

**Reason:** T002 successfully extracted the records and preserved their provenance. The problem was validation/classification, not extraction or transformation.

---

## D012 — Conversation Segmentation & Zero-Leakage Stratified Splitting

**Decision:** 4-hour temporal inactivity threshold segments chats into cohesive dialogue sessions; train/dev/test splits strictly partition at the conversation level with length-bin and initiation stratification.

**Reason:** Message-level random splitting causes severe context leakage across splits. Conversation-level isolation ensures holdout testing truly evaluates generalization.

---

## D013 — Sentence-BERT Model Selection for Hinglish Dialogue

**Decision:** Adopt `l3cube-pune/hindi-sentence-bert-nli` (768 dimensions, 12 layers) as the primary embedding model over generic multilingual MiniLM and raw BERT Masked-LMs.

**Reason:** Contrastive benchmarking revealed +57.1% wider semantic gap on conversational Hinglish triplets, isotropic unit-sphere projections ($L_2 = 1.0$), and strong representation of code/debugging vernacular.

---

## D014 — Vectorized Sub-millisecond Semantic Similarity & Diversity Index

**Decision:** Precompute normalized dot products for vector search; compute Response Diversity Index ($1 - \text{mean pairwise similarity}$) and Context-to-Response alignment across all training turns.

**Reason:** Allows sub-millisecond candidate retrieval ($<1$ ms on CPU) and provides an empirical baseline of conversational variety ($0.7330$).

---

## D015 — Spherical K-Means & Unsupervised Archetype Discovery

**Decision:** Perform K-Means on $L_2$-normalized vector space with centroid re-projection onto the unit hypersphere; evaluate $K \in [6, 18]$ via Cosine Silhouette and Calinski-Harabasz metrics; extract true dataset medoids ($m_k = \arg\max v_i \cdot \mu_k$) and compute class-based TF-IDF (c-TF-IDF) profiles.

**Reason:** Euclidean distance on unit vectors monotonically maps to cosine distance ($\|u - v\|^2 = 2 - 2(u \cdot v)$). The unsupervised sweep identified 6 distinct conversational archetypes covering 100% of training data without manual labeling.

---

## D016 — Explicit Separation of Topic from Style & Situational Probability Matrix

**Decision:** Isolate conversational topic discovery ('what is discussed') from communication style ('how it is spoken') by modeling incoming context embeddings ($K=11, \text{Silhouette}=0.2365$), aggressively filtering out conversational slang and chat participant names, and computing the $P(\text{Style} \mid \text{Topic})$ transition matrix.

**Reason:** Preserves the core architectural principle that professional topics (projects, resume, technical skills) do not dictate a single monolithic style, but systematically shift the probability distribution over response archetypes (e.g. Travel/Flight APIs triggers 53.0% Technical Collaboration, while Formal OJT prompts 24.6% Minimalist Confirmations).

---

## D017 — Quantitative Linguistic Constraints from Empirical Ground Truth

**Decision:** Formally measure and enforce low-level surface linguistic distributions (92.5% zero terminal punctuation, 44.4% multi-bubble burstiness, 43.5% all-lowercase text, 6.6% emoji density with 41.4% burst repetition, and 50.6% hapax legomena ratio) as hard constraints on generation.

**Reason:** Language models default to standard formal prose (e.g., ending every message with a period, sending single long paragraphs, capitalization). Hard constraints grounded in empirical statistics are essential to guarantee authentic persona replication.

---

## D018 — Code-Mixed Syntactic Clause Structuring & Speech Act Quantification

**Decision:** Explicitly model Hinglish syntactic structures: 25.4% verbless fragments (reaching 82.2% in confirmations and 54.7% in reactive slang), 57.4% simple independent clauses, only 3.6% complex subordination, 21.6% negation (reaching 91.5% in Denial/Friction), and a 2.66x self-to-other pronoun orientation ratio.

**Reason:** LLMs tend to over-generate complex subordinations ("Because X happened, although Y was true..."). Capturing the exact fragment vs simple clause ratio ensures the AI Spokesperson constructs dialogue with your genuine conversational rhythm.

---

## D019 — Empirical Code-Switching Ratios & Token Language Distributions

**Decision:** Quantify Hindi-English code-switching into 4 operational conversational modes: Pure English (5.3%), Pure Hindi (43.4%), Code-Switched Hinglish (48.2%), and Neutral/Media (3.1%), with an overall token language distribution of 27.2% English vs 72.8% Hindi, and style-dependent switching dynamics (ranging from 0.1 switches/turn in Minimalist Confirmations to 9.6 switches/turn in Technical Collab).

**Reason:** Persona generation cannot treat bilingualism as uniform noise. Technical discussions organically trigger frequent intra-sentential English insertions (38.5% English tokens, 88.4% code-switched turns), while emotional banter and friction lean heavily on Romanized Hindi matrices (80.7%–87.4% Hindi tokens). Imposing style-conditional code-switching guarantees authentic lexical insertion without awkward translation artifacts.

---

## D020 — Discourse Act Modeling & Empirical Context Transitions

**Decision:** Formally classify conversational turns into 10 pragmatic discourse functions (`AGREE` [9.9%], `DISAGREE` [15.3%], `QUESTION` [19.6%], `EXPLAIN` [1.4%], `CLARIFY` [0.6%], `HEDGE` [1.3%], `ADVICE` [1.7%], `HUMOR` [4.0%], `EMPHASIS` [1.5%], `INFORMATIVE` [44.9%]), and measure empirical context-conditioned transition probabilities $P(\text{Response Act} \mid \text{Context Act})$ (e.g., questions elicit 43.3% informative answers, 20.2% counter-questions, and 16.4% direct disagreement).

**Reason:** High-fidelity persona generation requires modeling interpersonal communicative intent, not just vocabulary or grammar. The empirical discourse profile reveals strong style-dependent pragmatic stances (e.g. 67.0% disagreement in Denial/Friction, 92.1% interrogative probing in Inquisitive Probing, and 38.9% agreement in Minimalist Confirmations) that instruct the downstream spokesperson when to push back, probe, explain, or confirm.

---

## D021 — Multi-Signal Situational Context Classification

**Decision:** Segment conversational environments into 6 core situations: Advice / Probing (39.3%), Casual Banter (19.9%), Career / Academic (18.4%), Technical Collab (12.3%), Conflict / Friction (6.3%), and Acknowledgement (3.7%), combining multi-turn lexical cues, topic clustering, and response brevity.

**Reason:** Communication personas do not exist in a vacuum; human communicators dramatically modulate tone, vocabulary, and syntax based on situational context. Grounding the persona in 6 distinct situational modes enables the spokesperson to switch naturally between terse technical problem solving (27.1% in C04), peer academic discussions, and casual emotional banter.

---

## D022 — Quantified Behavioral Profiling with Empirical Evidence Exemplars

**Decision:** Formulate 10 normalized psychological/behavioral dimensions ($[0.0, 1.0]$): Directness (0.45), Verbosity (0.31), Hedging (0.03), Disagreement Style (0.19), Humor/Playfulness (0.06), Question Initiative (0.23), Confidence/Assertion (0.48), Emotional Expressiveness (0.04), Code-Switching Propensity (0.33), and Technical Depth (0.03), grounded with top mined supporting exemplar pairs from the corpus.

**Reason:** Raw token statistics alone do not give language model system prompts actionable behavioral instructions. Quantifying explicit behavioral dimensions provides unambiguous calibration targets (e.g., extremely low hedging at 0.03 and high confidence at 0.48 prohibit apologetic sycophancy, while 0.80 disagreement in friction situations enforces authentic unvarnished pushback).

---

## D023 — Global & Situational Style Synthesis for Persona Delivery

**Decision:** Synthesize all empirical NLP analyses (linguistics, syntax, code-switching, discourse acts, and behavioral traits) into two complementary tiers:
1. **Global Style Profile (T022)**: Encapsulating overall core dimensions (Formality: 0.27, Directness: 0.45, Verbosity: 0.31, Confidence: 0.48, Hedging: 0.03), surface constraints (92.5% zero terminal punctuation, 43.5% lowercase, 19.9% multi-bubble bursts), and language mechanics (27.2% English vs 72.8% Hindi tokens, 48.2% code-switched).
2. **Situational Style Profiles (T023)**: Modulating these parameters dynamically across 6 conversational environments: Technical Collab (41.5% English tokens, 0.20 technical depth), Conflict / Friction (90.0% disagreement, 89.2% Hindi matrix), Acknowledgement (1.00 directness, 0.03 verbosity), Casual Banter, Career / Academic, and Advice / Probing.

**Reason:** A static single prompt cannot capture human conversational dynamics. Providing both a global baseline and situation-conditioned modulations enables the downstream AI Spokesperson to maintain authentic baseline habits while shifting smoothly between terse technical problem solving, peer banter, and direct disagreement.

---

## D024 — Strict Epistemic Partitioning in Structured Persona Inference

**Decision:** Formulate all higher-level behavioral interpretations with strict tripartite epistemic partitioning: `OBSERVED` (hard empirical facts derived from the corpus), `INFERRED` (conservative, testable generalizations about communicative stance), and `UNKNOWN` (private relationships, personal finances, unverified career plans, and emotional interiors).

**Reason:** Unconstrained LLMs frequently hallucinate biographical gossip or invent personality traits when summarizing chats. Explicitly demanding an `UNKNOWN` category forces the model to quarantine non-communication private domains and preserves privacy under the Core Decoupling Principle.

---

## D025 — Grounded Evidence Registry with Auditable Turn-Level Provenance

**Decision:** Maintain a centralized, queryable `EvidenceRegistry` where every behavioral rule and stylistic constraint in the persona package is bound to an empirical score, statistical confidence interval ($\ge 0.93$), epistemic status, and exact supporting turn counts (e.g. 2,542 turns for zero terminal punctuation, 1,324 turns for code-switching).

**Reason:** Eliminates "black box" persona generation. Every guideline injected into the downstream AI Spokesperson system prompt can be traced directly to an auditable empirical footprint in the sanitized dataset.

---




# 22. DOWNSTREAM INTEGRATION CONTRACT

The Persona Engine will eventually expose a clean interface to the AI Spokesperson.

Conceptually:

```text
Persona Engine
      |
      v
Persona Package
      |
      +--------------------------+
      |                          |
      v                          v
Static Persona Context      Style Example Index
      |                          |
      +-------------+------------+
                    |
                    v
             AI Spokesperson
```

The spokesperson should be able to obtain:

```text
get_global_style()
get_situational_style(context)
get_style_examples(context)
get_linguistic_constraints()
```

The Persona Engine must not own:

- portfolio facts;
- resume facts;
- project descriptions;
- professional links;
- frontend UI;
- voice;
- project-opening tools.

Those belong to the downstream spokesperson system.

---

# 23. FINAL SUCCESS CRITERIA

The project is considered complete when:

1. A WhatsApp export can be parsed reliably.
2. Messages are represented in a canonical schema.
3. Parser/transformation quality is automatically validated.
4. Private information can be sanitized.
5. Conversations are reconstructed contextually.
6. A high-quality analysis dataset is generated.
7. Transformer-based semantic representations are available.
8. Linguistic characteristics are quantified.
9. Discourse and behavioral characteristics are analyzed.
10. Situational communication styles are represented.
11. Persona inference produces structured, evidence-backed outputs.
12. Representative style examples are generated.
13. Style embeddings can be retrieved.
14. A complete Persona Package is generated.
15. The system is evaluated against held-out data.
16. Failure modes are documented.
17. Processing is reproducible and cached.
18. The resulting package can be consumed independently by the future AI spokesperson.

---

# 24. CURRENT TASK

## Current Implementation State

T001–T011 are complete.

The implemented foundation is:

```text
WhatsApp .txt
    ↓
T002 Parser [COMPLETE]
    ↓
T003 Canonical message dataset [COMPLETE]
    ↓
T004 Validation [COMPLETE]
    ↓
T005 Privacy [COMPLETE]
    ↓
T006 Cleaning / Text Normalization [COMPLETE]
    ↓
T007 Temporal Conversation Segmentation [COMPLETE]
    ↓
T008 Context Reconstruction [COMPLETE]
    ↓
T009 Quality Filtering [COMPLETE]
    ↓
T010 Sampling Strategy [COMPLETE]
    ↓
T011 Train / Development / Holdout Split [COMPLETE]
```

Current partitioned dataset (zero conversation leakage verified):

```text
Train: 2,750 pairs (642 conversations, ~70%)
Dev: 594 pairs (140 conversations, ~15%)
Test (Holdout): 612 pairs (140 conversations, ~15%)
Zero leakage verified: True
0 missing senders / senders strictly preserved
0 schema errors
0 provenance errors
```

### Completed Phase
- **T012 — Transformer Embedding Layer**: Implemented in `src/nlp/embeddings.py`, verified with 62/62 tests passing. Stored `train_target_vectors.npz` (7.5 MB), `train_context_vectors.npz` (7.3 MB), and metadata in `data/processed/embeddings/` using `l3cube-pune/hindi-sentence-bert-nli` (768 dimensions, L2 normalized, 0 NaNs).
- **T013 — Semantic Similarity**: Implemented in `src/nlp/similarity.py`, verified with 68/68 tests passing. Analyzed 2,750 training pairs: Response Diversity Index ($0.7330$), Context-Response Alignment breakdown (8.4% mirroring, 75.1% balanced, 16.5% reactive), and vectorized sub-millisecond Top-K retrieval. Stored `data/processed/similarity_report.json`.
-**T014 — Semantic Clustering**: Implemented in `src/nlp/clustering.py`, verified with 75/75 tests passing. Executed Spherical K-Means evaluation across $K \in [6, 18]$ on 2,750 training turns. Discovered optimal resolution $K = 6$ (Silhouette = 0.0610, CH = 103.16) uncovering 6 clear texting archetypes: Quick Reactive (23.1%), Extended Venting/Banter (22.6%), Technical/Project Collab (19.1%), Denial/Reassurance (14.6%), Minimalist Confirmations (11.0%), and Direct Inquisitive Probing (9.6%). Stored `cluster_assignments.jsonl`, `cluster_profiles.json`, and `clustering_report.json`.
- **T015 — Topic Discovery & Tagging**: Implemented in `src/nlp/topics.py`, verified with 80/80 tests passing. Modeled context embeddings across $K \in [5, 11]$ ($K=11, \text{Silhouette}=0.2365, \text{CH}=124.17$), discovering 11 domain topics (LaTeX Resume Formatting, Flight/Travel APIs, Hotel APIs, XML Backend Serialization, College Placements & Companies, Exams/Academics, Formal OJT). Discovered the empirical Situational Transition Probability Matrix $P(\text{Style} \mid \text{Topic})$ proving topic-driven style modulation (e.g., Flight APIs triggers 53.0% Technical Collab, while Formal OJT triggers 24.6% Minimalist Confirmations). Stored `topic_assignments.jsonl`, `topic_profiles.json`, `topic_style_matrix.json`, and `topic_report.json`.
- **T016 — Surface Linguistic Profile**: Implemented in `src/nlp/linguistics.py`, verified with 87/87 tests passing. Measured empirical distributions across 2,750 training turns: 92.5% zero terminal punctuation, 44.4% multi-bubble burstiness (avg 1.86 msgs/turn), 43.5% all-lowercase casing, 6.6% selective emoji usage (41.4% burst repetition, #1 emoji 😭 with 147 occurrences), and 50.6% hapax legomena ratio. Stored `global_linguistic_profile.json`, `style_linguistic_profiles.json`, and `linguistics_report.json`.
- **T017 — Syntactic Profile**: Implemented in `src/nlp/syntax.py`, verified with 94/94 tests passing. Quantified clause structures and speech acts across code-mixed Hinglish: 25.4% verbless fragments (reaching 82.2% in confirmations and 54.7% in reactive slang), 57.4% simple clauses, only 3.6% complex subordination, 23.2% interrogatives (surging to 93.2% in probing), 21.6% negation (reaching 91.5% in Denial/Friction), and a 2.66x self-to-other pronoun orientation ratio. Stored `global_syntactic_profile.json`, `style_syntactic_profiles.json`, and `syntax_report.json`.

### Next task

Proceed to:

```text
T018 Code-Switching / Language Analysis
```





Then continue in dependency order through semantic similarity, clustering, topic discovery, linguistic fingerprinting, behavioral inference, persona packaging, and evaluation.


---

# 25. CONTEXT-RECOVERY INSTRUCTION

If this document is supplied in a future conversation, treat it as the authoritative project specification unless the user explicitly requests a change.

Before proposing implementation changes:

1. Identify the relevant task ID.
2. Check the existing architecture and Decision Log.
3. Do not contradict established decisions silently.
4. If a change is necessary, explain why.
5. Update the Decision Log before treating the new approach as final.
6. Keep the work focused on the Persona Extraction Engine unless the user explicitly switches scope.

The project is NOT to be reframed as a V1/V2/MVP roadmap. It is one complete system implemented in dependency order.

The eventual objective remains:

**WhatsApp → modern NLP/ML analysis → evidence-backed communication persona → reusable Persona Package → AI Spokesperson integration.**
