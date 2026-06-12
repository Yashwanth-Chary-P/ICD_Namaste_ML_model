# Project Analysis

## Purpose

This repository contains data preparation, retrieval experiments, and error analysis for mapping NAMASTE/Ayurveda terminology to ICD/TM2-style medical classification codes. The project is data-first: most scripts read CSV inputs from their local folder, transform or evaluate records, and write CSV outputs for downstream inspection.

## Main Workflows

1. ICD/TM2 preprocessing
   - `icd-preprocessing/` and `tm2-preprocessing/` contain scripts and CSV inputs used to merge, clean, and finalize ICD/TM2 code tables.
   - The important outputs include `tm2_final.csv`, `tm2.csv`, and merged ICD CSV files.

2. NAMASTE preprocessing
   - `namaste-preprocessing/` cleans Ayurveda, Siddha, and Unani source files.
   - `namaste-preprocessing-validate/` deduplicates Ayurveda data, extracts TM2 mappings, and creates evaluation-ready datasets.

3. Model experiments
   - `ML model/` contains early and comparative retrieval experiments.
   - Models include TF-IDF, BM25, optional BERT embeddings, and hybrid scoring.

4. Research and error analysis
   - `research/` contains the current evaluation pipeline, ablation studies, qualitative examples, and deeper error-analysis exports.
   - Outputs under `research/outputs_error_analysis/`, `research/deep_error_analysis_outputs/`, and related folders document model behavior and failure modes.

5. Backend data preparation
   - `backend data/` contains processed CSV assets intended for downstream application or service use.

## Dependencies

The project uses Python with these primary libraries:

- `pandas` and `numpy` for tabular processing.
- `scikit-learn` for TF-IDF, cosine similarity, and train/test splits.
- `rank-bm25` for BM25 retrieval baselines.
- `sentence-transformers` and `torch` for optional BERT experiments.
- `scipy` for correlation analysis.
- `tqdm` for progress indicators.

## Current Architecture

The codebase is organized as script-based pipelines rather than a packaged Python application. Most scripts assume the current working directory contains their input CSV files. For example, the research scripts should be run from inside `research/`, while preprocessing scripts should be run from their respective folders.

## Preservation Notes

No functionality has been intentionally removed. The improvements focus on repository hygiene, documentation, reproducible setup, validation support, and safer error handling in analysis scripts.

