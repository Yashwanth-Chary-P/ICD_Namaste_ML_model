# Architecture

## Repository Map

| Path | Role |
| --- | --- |
| `bin/` | Early ICD data-fetching and parsing scripts with source chapter CSVs. |
| `icd-preprocessing/` | ICD chapter data, merge scripts, and finalized ICD CSV outputs. |
| `tm2-preprocessing/` | TM2 preprocessing script and final TM2 datasets. |
| `namaste-preprocessing/` | NAMASTE source preprocessing for Ayurveda, Siddha, and Unani datasets. |
| `namaste-preprocessing-validate/` | Deduplication, TM2 mapping extraction, and evaluation dataset creation. |
| `ML model/` | Model comparison experiments and prediction exports. |
| `research/` | Current retrieval, ablation, qualitative, and error-analysis workflows. |
| `backend data/` | Processed datasets prepared for backend or downstream consumption. |
| `final/` | Finalized preprocessing artifact snapshots. |
| `testing_data/` | Small test/evaluation data preparation scripts and CSVs. |

## Data Flow

```text
Source chapter and NAMASTE CSVs
        |
        v
ICD/TM2 preprocessing + NAMASTE preprocessing
        |
        v
Cleaned Ayurveda and TM2 evaluation datasets
        |
        v
Retrieval experiments: TF-IDF, BM25, BERT, Hybrid
        |
        v
Prediction CSVs, comparison summaries, ablations
        |
        v
Deep error analysis and qualitative examples
```

## Execution Model

Scripts are folder-local. Run a script from the directory that contains the CSV files referenced in its config block:

```powershell
cd research
python error_analysis.py
python error_anlaysis_data.py
python qualitative_examples.py
```

The optional BERT workflows require `sentence-transformers` and a working PyTorch install. If those packages are unavailable, scripts that guard BERT imports continue with sparse retrieval models.

