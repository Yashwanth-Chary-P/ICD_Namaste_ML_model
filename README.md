# ICD NAMASTE ML Model

This repository contains preprocessing pipelines, retrieval experiments, evaluation datasets, and error-analysis outputs for mapping NAMASTE traditional medicine terminology to ICD/TM2-style medical classification codes.

The project focuses on Ayurveda terminology and compares sparse, dense, hybrid, and reranked retrieval strategies using curated CSV datasets.

## What This Project Does

- Cleans and prepares ICD/TM2 and NAMASTE source datasets.
- Builds evaluation datasets with NAMASTE codes mapped to TM2 labels.
- Runs retrieval models including TF-IDF, BM25, optional BERT embeddings, and hybrid scoring.
- Produces top-k prediction files and model comparison summaries.
- Performs deep error analysis for false positives, ambiguous queries, recoverable ranking failures, and attractor documents.
- Exports backend-ready processed CSV datasets.

## Repository Structure

| Path | Description |
| --- | --- |
| `bin/` | Early ICD chapter parsing and data-fetching scripts. |
| `icd-preprocessing/` | ICD chapter data, merge scripts, and finalized ICD CSV files. |
| `tm2-preprocessing/` | TM2 preprocessing script and final TM2 CSVs. |
| `namaste-preprocessing/` | Source preprocessing for Ayurveda, Siddha, and Unani datasets. |
| `namaste-preprocessing-validate/` | Deduplication and evaluation dataset generation. |
| `ML model/` | Retrieval experiments, model comparisons, and prediction outputs. |
| `research/` | Current error analysis, ablation studies, qualitative examples, and result exports. |
| `backend data/` | Processed datasets intended for downstream backend use. |
| `final/` | Final preprocessing artifact snapshots. |
| `testing_data/` | Small evaluation and testing data preparation scripts. |
| `docs/` | Architecture, analysis, and workflow documentation. |
| `scripts/` | Repository validation utilities. |

## Requirements

- Python 3.10 or newer is recommended.
- A virtual environment is strongly recommended because optional dense retrieval dependencies can be large.

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The BERT-based experiments require `sentence-transformers` and `torch`. Scripts that mark BERT as optional continue to run sparse retrieval experiments when those packages are unavailable.

## Quick Validation

From the repository root:

```powershell
python scripts/validate_project.py
```

This checks expected project files, parses Python syntax without writing bytecode, and reports missing required or optional packages.

## Typical Workflows

Run scripts from their own folders because most scripts use folder-local CSV paths.

Create validated NAMASTE evaluation datasets:

```powershell
cd namaste-preprocessing-validate
python main.py
```

Run the main research error-analysis pipeline:

```powershell
cd research
python error_analysis.py
python error_anlaysis_data.py
python qualitative_examples.py
```

Run model comparison experiments:

```powershell
cd "ML model"
python researchWithoutDictionary.py
python researchWithoutDictionaryWithLabel.py
python researchWithoutDictionaryWithLabel_bm25.py
```

## Modeling Approach

The retrieval experiments build query text from NAMASTE terms and definitions, then compare those queries against TM2 title and index-term text.

Implemented approaches include:

- TF-IDF cosine similarity baselines.
- BM25 lexical retrieval baselines.
- Optional SentenceTransformer dense retrieval.
- Hybrid sparse plus dense scoring.
- Reranking boosts based on Sanskrit or token evidence.

## Outputs

Important generated outputs include:

- `research/outputs_error_analysis/comparison_summary.csv`
- `research/outputs_error_analysis/all_top1_errors.csv`
- `research/outputs_error_analysis/predictions_*.csv`
- `research/deep_error_analysis_outputs/*.csv`
- `research/qualitative_example_outputs/*.csv`
- `ML model/comparison_summary.csv`
- `ML model/predictions_*.csv`

## Documentation

Additional documentation is available in:

- `docs/PROJECT_ANALYSIS.md`
- `docs/ARCHITECTURE.md`
- `docs/DATASETS_AND_WORKFLOWS.md`

## Notes

The repository is currently script-oriented rather than packaged as an importable Python module. Preserving folder-local execution keeps the existing workflows compatible with the CSV paths already used throughout the project.

