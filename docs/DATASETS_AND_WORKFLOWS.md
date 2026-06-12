# Datasets and Workflows

## Folder-local Scripts

Most scripts use relative CSV paths. Run them from the directory where the script lives unless the script explicitly states otherwise.

```powershell
cd namaste-preprocessing-validate
python main.py

cd ..\research
python error_analysis.py
python error_anlaysis_data.py
python qualitative_examples.py
```

## Important Dataset Groups

| Dataset group | Typical location | Description |
| --- | --- | --- |
| ICD chapters | `icd-preprocessing/data/` | Raw and merged ICD chapter data. |
| TM2 code tables | `tm2-preprocessing/`, `research/`, `ML model/` | TM2 terminology and final evaluation tables. |
| NAMASTE sources | `namaste-preprocessing/` | Ayurveda, Siddha, and Unani source CSVs. |
| Evaluation data | `namaste-preprocessing-validate/`, `research/` | Cleaned Ayurveda rows with mapped TM2 labels. |
| Prediction outputs | `ML model/`, `research/outputs_error_analysis/` | Top-k predictions and comparison summaries. |
| Error-analysis outputs | `research/deep_error_analysis_outputs/`, `research/qualitative_example_outputs/` | Failure summaries, false-positive hubs, qualitative cases, and recovery analysis. |

## Main Experiment Types

1. Sparse lexical retrieval
   - TF-IDF with n-grams and cosine similarity.
   - BM25 over tokenized TM2 document text.

2. Optional dense retrieval
   - SentenceTransformer embeddings using `all-MiniLM-L6-v2`.
   - Automatically skipped in guarded scripts when `sentence-transformers` is unavailable.

3. Hybrid and reranking
   - Hybrid scoring combines normalized sparse and dense scores.
   - Reranking boosts candidates with direct Sanskrit or token evidence.

4. Error analysis
   - Top-k prediction exports are converted into model-level error summaries.
   - Deep analysis measures false-positive attractors, query ambiguity, recoverable errors, and document-length correlations.

## Validation

Run the repository validation helper from the project root:

```powershell
python scripts/validate_project.py
```

The helper checks required paths, parses Python syntax without writing `__pycache__`, and reports required or optional dependency gaps.

