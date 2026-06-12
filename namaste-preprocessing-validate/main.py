import pandas as pd
import re

# =========================================================
# INPUT / OUTPUT FILES
# =========================================================
INPUT_FILE = "AYURVEDA.csv"

OUTPUT_MASTER = "ayurveda_with_tm2_clean.csv"
OUTPUT_EVAL = "eval_dataset_final.csv"
OUTPUT_AYURVEDA = "ayurveda_clean_dataset.csv"
OUTPUT_DUPLICATES = "duplicate_rows.csv"
OUTPUT_COUNTS = "split_counts.csv"

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_FILE, encoding="utf-8")

# standardize columns
df.columns = df.columns.str.lower().str.strip()

print("📥 Loaded dataset")
print("📊 Shape:", df.shape)

# =========================================================
# SAFE TEXT
# =========================================================
def safe_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

# =========================================================
# NORMALIZE TEXT
# =========================================================
def normalize_spaces(text):

    text = safe_text(text)

    # normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# =========================================================
# NORMALIZE CODE
# =========================================================
def normalize_code(code):

    code = safe_text(code)

    # remove extra spaces
    code = re.sub(r"\s+", "", code)

    return code

# =========================================================
# APPLY BASIC CLEANING
# =========================================================
norm = df.copy()

for col in norm.columns:

    if col == "sr no.":
        continue

    norm[col] = norm[col].apply(normalize_spaces)

if "namc_code" in norm.columns:
    norm["namc_code"] = norm["namc_code"].apply(normalize_code)

# =========================================================
# FIND DUPLICATES
# =========================================================
duplicate_subset = [c for c in norm.columns if c != "sr no."]

duplicate_mask = norm.duplicated(
    subset=duplicate_subset,
    keep=False
)

duplicate_rows = df.loc[duplicate_mask].copy()

duplicate_groups = (
    norm.loc[duplicate_mask, duplicate_subset]
    .drop_duplicates()
    .shape[0]
)

duplicate_count = len(duplicate_rows)

# save duplicate rows
duplicate_rows.to_csv(
    OUTPUT_DUPLICATES,
    index=False,
    encoding="utf-8"
)

print("\n✅ Duplicate detection completed")
print("Duplicate rows:", duplicate_count)
print("Duplicate groups:", duplicate_groups)

# =========================================================
# REMOVE DUPLICATES
# =========================================================
keep_mask = ~norm.duplicated(
    subset=duplicate_subset,
    keep="first"
)

df_clean = df.loc[keep_mask].copy()

print("\n✅ Duplicate removal completed")
print("Rows after deduplication:", len(df_clean))

# =========================================================
# IDENTIFY TM2 ROWS
# =========================================================
# IMPORTANT RULE:
#
# TM2 rows are identified ONLY using:
# "Name English" contains TM2
#
# NOT using NAMC_CODE prefix.
#
# This avoids false positives like:
# SA, SB, SC etc.
# =========================================================
tm2_mask = df_clean["name english"] \
    .astype(str) \
    .str.contains(r"\btm2\b", case=False, na=False)

df_clean["is_tm2"] = tm2_mask.astype(int)

# =========================================================
# SEPARATE TM2 CODE + AYURVEDA CODE
# =========================================================
def extract_codes(code):

    code = normalize_code(code)

    tm2_code = ""
    ayurveda_code = ""

    # -----------------------------------------------------
    # CASE:
    # SR11(AAA-1)
    # SM75(AAC-14)
    # AAC-239(SK30)
    # AAC-25(SP9Y)
    # -----------------------------------------------------
    pattern = r"^([A-Z0-9\-\.]+)\(([A-Z0-9\-\.]+)\)$"

    match = re.match(pattern, code)

    if match:

        first = match.group(1).strip()
        second = match.group(2).strip()

        # ---------------------------------------------
        # TM2 first
        # Example:
        # SR11(AAA-1)
        # ---------------------------------------------
        if first.startswith("S"):

            tm2_code = first
            ayurveda_code = second

        # ---------------------------------------------
        # Ayurveda first
        # Example:
        # AAC-239(SK30)
        # ---------------------------------------------
        else:

            ayurveda_code = first
            tm2_code = second

    else:

        # ---------------------------------------------
        # NO BRACKET CASE
        # ---------------------------------------------
        if code.startswith("S"):

            tm2_code = code

        else:

            ayurveda_code = code

    return pd.Series([tm2_code, ayurveda_code])

# apply extraction
df_clean[["tm2_code", "ayurveda_code"]] = (
    df_clean["namc_code"]
    .astype(str)
    .apply(extract_codes)
)

# =========================================================
# SPLIT DATASETS
# =========================================================
tm2_df = df_clean[df_clean["is_tm2"] == 1].copy()

ayu_df = df_clean[df_clean["is_tm2"] == 0].copy()

# =========================================================
# SAVE TM2 EVALUATION DATASET
# =========================================================
tm2_cols = [
    "sr no.",
    "namc_id",
    "namc_code",
    "tm2_code",
    "ayurveda_code",
    "namc_term",
    "namc_term_diacritical",
    "namc_term_devanagari",
    "short_definition",
    "long_definition",
    "ontology_branches",
    "name english",
    "name english under index",
    "primary index related",
    "is_tm2"
]

tm2_cols = [c for c in tm2_cols if c in tm2_df.columns]

tm2_final = tm2_df[tm2_cols].copy()

tm2_final.to_csv(
    OUTPUT_EVAL,
    index=False,
    encoding="utf-8"
)

# =========================================================
# SAVE PURE AYURVEDA DATASET
# =========================================================
ayu_final = ayu_df.copy()

ayu_final.to_csv(
    OUTPUT_AYURVEDA,
    index=False,
    encoding="utf-8"
)

# =========================================================
# SAVE MASTER CLEAN DATASET
# =========================================================
df_clean.to_csv(
    OUTPUT_MASTER,
    index=False,
    encoding="utf-8"
)

# =========================================================
# SAVE COUNTS
# =========================================================
counts_df = pd.DataFrame([{
    "original_rows": len(df),
    "duplicate_rows": duplicate_count,
    "duplicate_groups": duplicate_groups,
    "rows_after_deduplication": len(df_clean),
    "tm2_rows": len(tm2_df),
    "ayurveda_rows": len(ayu_df)
}])

counts_df.to_csv(
    OUTPUT_COUNTS,
    index=False,
    encoding="utf-8"
)

# =========================================================
# FINAL REPORT
# =========================================================
print("\n✅ ALL PROCESSING COMPLETED")

print("\n📂 FILES CREATED")
print("----------------------------------")
print("1.", OUTPUT_MASTER)
print("2.", OUTPUT_EVAL)
print("3.", OUTPUT_AYURVEDA)
print("4.", OUTPUT_DUPLICATES)
print("5.", OUTPUT_COUNTS)

print("\n📊 FINAL COUNTS")
print("----------------------------------")
print("Original rows:", len(df))
print("Duplicate rows:", duplicate_count)
print("Duplicate groups:", duplicate_groups)
print("Rows after deduplication:", len(df_clean))
print("TM2 rows:", len(tm2_df))
print("Ayurveda rows:", len(ayu_df))

print("\n🔍 SAMPLE CODE EXTRACTION")
print(
    df_clean[[
        "namc_code",
        "tm2_code",
        "ayurveda_code",
        "is_tm2"
    ]]
    .head(10)
    .to_string(index=False)
)