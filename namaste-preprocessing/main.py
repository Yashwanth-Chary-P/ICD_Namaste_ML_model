import pandas as pd
import re
import unicodedata

# ==============================
# LOAD
# ==============================
df = pd.read_csv("AYURVEDA.csv", dtype=str, keep_default_na=False, encoding="utf-8")
df.columns = df.columns.str.lower().str.strip()

print("📥 Total rows (original):", len(df))

# ==============================
# NORMALIZE TEXT
# ==============================
def norm(x):
    x = "" if x is None else str(x)
    x = unicodedata.normalize("NFKC", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

for col in df.columns:
    df[col] = df[col].map(norm)

# Keep a raw copy of original NAMC_CODE for reference
df["namc_code_raw"] = df["namc_code"]

# ==============================
# TM2 CODE EXTRACTION
# Valid TM2 pattern:
# S + letters/digits, and MUST contain at least one digit
# Excludes: S, SA, SB, SC ...
# Examples kept: SR11, SQ00, SP9Y, SK84, S1B
# ==============================
tm2_pattern = re.compile(r"S(?=[A-Z0-9]*\d)[A-Z0-9]+")

def extract_tm2(code):
    code = norm(code)

    if code in {"S", "SA"}:
        return ""

    m = tm2_pattern.search(code)
    return m.group(0) if m else ""

# ==============================
# AYUSH CODE EXTRACTION
# Keep original Ayurvedic code only
# Remove TM2 part if it is mixed with Ayurveda code
# ==============================
def extract_ayush(code):
    code = norm(code)

    if code in {"S", "SA"}:
        return code

    # remove TM2 token if present
    code = tm2_pattern.sub("", code)

    # remove brackets and extra spaces
    code = re.sub(r"[()]", " ", code)
    code = re.sub(r"\s+", " ", code).strip()

    return code

df["tm2_code"] = df["namc_code"].apply(extract_tm2)
df["namc_code"] = df["namc_code"].apply(extract_ayush)

# ==============================
# COUNT BEFORE DEDUP
# ==============================
total_rows_before = len(df)
tm2_rows_before = (df["tm2_code"] != "").sum()

# ==============================
# DUPLICATES
# Remove only exact duplicate records, but keep one copy.
# We do NOT collapse different Ayurveda codes that map to same TM2 code.
# ==============================
dedup_subset = [c for c in df.columns if c not in {"sr no.", "namc_id", "namc_code_raw"}]

duplicates = df[df.duplicated(subset=dedup_subset, keep=False)].copy()
print("🔁 Duplicate rows found:", len(duplicates))

# Keep one copy of duplicates
df_clean = df.drop_duplicates(subset=dedup_subset, keep="first").copy()

# ==============================
# SAVE FILES
# ==============================
df_clean.to_csv("ayurveda_with_tm2_clean.csv", index=False, encoding="utf-8")
duplicates.to_csv("duplicate_rows.csv", index=False, encoding="utf-8")

# ==============================
# COUNTS AFTER DEDUP
# ==============================
print("\n✅ DONE")
print("Total rows before:", total_rows_before)
print("Rows with TM2 before dedup:", tm2_rows_before)
print("Rows after dedup:", len(df_clean))
print("Rows with TM2 after dedup:", (df_clean["tm2_code"] != "").sum())

print("\n🔍 Sample:")
print(df_clean[["namc_code", "tm2_code"]].head(10).to_string(index=False))