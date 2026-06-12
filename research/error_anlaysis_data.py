import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

TM2_FILE = "tm2.csv"
ERROR_FILE = "outputs_error_analysis/all_top1_errors.csv"

OUTPUT_DIR = Path("deep_error_analysis_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

print("\n📥 Loading files...")

tm2 = pd.read_csv(TM2_FILE)
errors = pd.read_csv(ERROR_FILE)

tm2.columns = [c.strip().lower().replace(" ", "_") for c in tm2.columns]

print("TM2 rows:", len(tm2))
print("Error rows:", len(errors))

# =========================================================
# AUTO-DETECT COLUMNS
# =========================================================

possible_code_cols = ["code", "tm2_code"]
possible_title_cols = ["title", "tm2_title"]
possible_index_cols = ["index_terms", "index_term"]

CODE_COL = None
TITLE_COL = None
INDEX_COL = None

for c in possible_code_cols:
    if c in tm2.columns:
        CODE_COL = c

for c in possible_title_cols:
    if c in tm2.columns:
        TITLE_COL = c

for c in possible_index_cols:
    if c in tm2.columns:
        INDEX_COL = c

print("\n🔎 Detected Columns")
print("CODE_COL :", CODE_COL)
print("TITLE_COL:", TITLE_COL)
print("INDEX_COL:", INDEX_COL)


def require_column(name, value, candidates):
    if value is None:
        raise ValueError(
            f"Missing required {name} column. Expected one of: {', '.join(candidates)}"
        )
    return value


CODE_COL = require_column("TM2 code", CODE_COL, possible_code_cols)
TITLE_COL = require_column("TM2 title", TITLE_COL, possible_title_cols)
INDEX_COL = require_column("TM2 index terms", INDEX_COL, possible_index_cols)

required_error_cols = [
    "top1_pred",
    "error_type",
    "model",
    "true_rank",
    "query_sanskrit_term",
]
missing_error_cols = [col for col in required_error_cols if col not in errors.columns]
if missing_error_cols:
    raise ValueError(
        "Missing required error analysis columns: " + ", ".join(missing_error_cols)
    )

# =========================================================
# TERM COUNTS
# =========================================================

def count_terms(x):
    if pd.isna(x):
        return 0
    return len(str(x).split(";"))

tm2["num_index_terms"] = tm2[INDEX_COL].apply(count_terms)

tm2["doc_length_words"] = (
    tm2[TITLE_COL].fillna("").astype(str)
    + " "
    + tm2[INDEX_COL].fillna("").astype(str)
).str.split().apply(len)

# =========================================================
# TOP 20 BIGGEST DOCUMENTS
# =========================================================

top20 = tm2.sort_values(
    "num_index_terms",
    ascending=False
)[[
    CODE_COL,
    TITLE_COL,
    "num_index_terms",
    "doc_length_words"
]].head(20)

print("\n=== TOP 20 TM2 DOCUMENTS ===")
print(top20)

top20.to_csv(
    OUTPUT_DIR / "top20_tm2_documents.csv",
    index=False
)

# =========================================================
# FALSE POSITIVE ANALYSIS
# =========================================================

fp_counts = (
    errors["top1_pred"]
    .value_counts()
    .reset_index()
)

fp_counts.columns = [
    "tm2_code",
    "false_positive_count"
]

merged_fp = fp_counts.merge(
    tm2[[CODE_COL, TITLE_COL, "num_index_terms", "doc_length_words"]],
    left_on="tm2_code",
    right_on=CODE_COL,
    how="left"
)

print("\n=== TOP FALSE POSITIVE DOCUMENTS ===")
print(merged_fp.head(20))

merged_fp.to_csv(
    OUTPUT_DIR / "top_false_positive_documents.csv",
    index=False
)

# =========================================================
# CORRELATION ANALYSIS
# =========================================================

merged_corr = tm2.merge(
    fp_counts,
    left_on=CODE_COL,
    right_on="tm2_code",
    how="left"
)

merged_corr["false_positive_count"] = (
    merged_corr["false_positive_count"]
    .fillna(0)
)

pearson_terms = pearsonr(
    merged_corr["num_index_terms"],
    merged_corr["false_positive_count"]
)

spearman_terms = spearmanr(
    merged_corr["num_index_terms"],
    merged_corr["false_positive_count"]
)

pearson_length = pearsonr(
    merged_corr["doc_length_words"],
    merged_corr["false_positive_count"]
)

spearman_length = spearmanr(
    merged_corr["doc_length_words"],
    merged_corr["false_positive_count"]
)

corr_df = pd.DataFrame({
    "metric": [
        "pearson_terms",
        "spearman_terms",
        "pearson_doc_length",
        "spearman_doc_length"
    ],
    "correlation": [
        pearson_terms[0],
        spearman_terms.correlation,
        pearson_length[0],
        spearman_length.correlation
    ],
    "p_value": [
        pearson_terms[1],
        spearman_terms.pvalue,
        pearson_length[1],
        spearman_length.pvalue
    ]
})

print("\n=== CORRELATION ANALYSIS ===")
print(corr_df)

corr_df.to_csv(
    OUTPUT_DIR / "correlation_analysis.csv",
    index=False
)

# =========================================================
# ERROR TYPE DISTRIBUTION
# =========================================================

error_types = (
    errors["error_type"]
    .value_counts()
    .reset_index()
)

error_types.columns = [
    "error_type",
    "count"
]

print("\n=== ERROR TYPE DISTRIBUTION ===")
print(error_types)

error_types.to_csv(
    OUTPUT_DIR / "error_type_distribution.csv",
    index=False
)

# =========================================================
# MODEL-WISE ERROR DISTRIBUTION
# =========================================================

model_errors = (
    errors["model"]
    .value_counts()
    .reset_index()
)

model_errors.columns = [
    "model",
    "error_count"
]

print("\n=== MODEL-WISE ERRORS ===")
print(model_errors)

model_errors.to_csv(
    OUTPUT_DIR / "model_error_distribution.csv",
    index=False
)

# =========================================================
# RECOVERABLE ERRORS
# =========================================================

recoverable = errors[
    (errors["true_rank"] > 1) &
    (errors["true_rank"] <= 5)
]

not_recoverable = errors[
    (errors["true_rank"].isna()) |
    (errors["true_rank"] > 5)
]

recovery_stats = []

overall_pct = (
    len(recoverable) / len(errors)
) * 100

recovery_stats.append({
    "model": "OVERALL",
    "recoverable_errors": len(recoverable),
    "total_errors": len(errors),
    "recovery_pct": overall_pct
})

for model in errors["model"].unique():

    sub = errors[
        errors["model"] == model
    ]

    rec = sub[
        (sub["true_rank"] > 1) &
        (sub["true_rank"] <= 5)
    ]

    pct = (
        len(rec) / len(sub)
    ) * 100

    recovery_stats.append({
        "model": model,
        "recoverable_errors": len(rec),
        "total_errors": len(sub),
        "recovery_pct": pct
    })

recovery_df = pd.DataFrame(recovery_stats)

print("\n=== RECOVERY ANALYSIS ===")
print(recovery_df)

recovery_df.to_csv(
    OUTPUT_DIR / "recovery_analysis.csv",
    index=False
)

# =========================================================
# QUERY AMBIGUITY ANALYSIS
# =========================================================

ambiguity = (
    errors.groupby("query_sanskrit_term")["top1_pred"]
    .nunique()
    .reset_index()
)

ambiguity.columns = [
    "query_term",
    "num_unique_predictions"
]

ambiguity = ambiguity.sort_values(
    "num_unique_predictions",
    ascending=False
)

print("\n=== QUERY AMBIGUITY ===")
print(ambiguity.head(30))

ambiguity.to_csv(
    OUTPUT_DIR / "query_ambiguity_analysis.csv",
    index=False
)

# =========================================================
# MOST ERROR-PRONE TERMS
# =========================================================

error_terms = (
    errors["query_sanskrit_term"]
    .value_counts()
    .reset_index()
)

error_terms.columns = [
    "query_term",
    "error_count"
]

print("\n=== MOST ERROR-PRONE TERMS ===")
print(error_terms.head(30))

error_terms.to_csv(
    OUTPUT_DIR / "most_error_prone_terms.csv",
    index=False
)

# =========================================================
# ATTRACTOR ANALYSIS
# =========================================================

top_attractors = merged_corr.sort_values(
    "false_positive_count",
    ascending=False
)[[
    CODE_COL,
    TITLE_COL,
    "num_index_terms",
    "doc_length_words",
    "false_positive_count"
]].head(30)

print("\n=== TOP ATTRACTOR DOCUMENTS ===")
print(top_attractors)

top_attractors.to_csv(
    OUTPUT_DIR / "top_attractor_documents.csv",
    index=False
)

# =========================================================
# SAVE COMPLETE MERGED DATA
# =========================================================

merged_corr.to_csv(
    OUTPUT_DIR / "full_tm2_analysis.csv",
    index=False
)

# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n✅ ALL ANALYSIS COMPLETED")

print("\n📁 Outputs saved in:")
print(OUTPUT_DIR)

print("\nIMPORTANT FILES TO SEND:")
print("1. correlation_analysis.csv")
print("2. recovery_analysis.csv")
print("3. query_ambiguity_analysis.csv")
print("4. top_attractor_documents.csv")
print("5. top_false_positive_documents.csv")
print("6. error_type_distribution.csv")
