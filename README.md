# Persona Project

A modular pipeline for persona generation and analysis.

## Project Structure

```text
persona/
│
├── data/
│   ├── raw/           # Original, immutable data dumps
│   ├── processed/     # Cleaned and transformed datasets
│   └── output/        # Generated persona artifacts, reports, and results
│
├── src/
│   ├── parser/        # Raw data extractors and parsers
│   ├── preprocessing/ # Data cleaning, normalization, and preparation
│   ├── segmentation/  # User segmentation and clustering algorithms
│   ├── nlp/           # NLP modeling, embeddings, and topic extraction
│   ├── persona/       # Persona profile synthesis and formatting
│   └── pipeline.py    # Main end-to-end execution pipeline
│
├── tests/             # Unit and integration tests
│
├── .gitignore         # Git ignore rules
├── requirements.txt   # Python project dependencies
└── README.md          # Project documentation
```

## Getting Started

1. **Virtual Environment Setup**:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Pipeline**:
   ```bash
   python src/pipeline.py
   ```
