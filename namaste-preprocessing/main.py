import pandas as pd
import re
import unicodedata

# =========================================================
# INPUT / OUTPUT
# =========================================================
INPUT_FILE = "AYURVEDA.csv"
OUTPUT_FILE = "AYURVEDA_processed.csv"

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_FILE, encoding="utf-8")
df.columns = df.columns.str.lower().str.strip()

print("📥 Loaded dataset")
print("Columns:", df.columns.tolist())

# =========================================================
# SAFE TEXT HANDLER
# =========================================================
def safe(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

# =========================================================
# CLEAN TEXT (LIKE ICD BUT SAFE FOR SANSKRIT)
# =========================================================
def clean_text(text):
    text = safe(text)

    # preserve unicode characters
    text = unicodedata.normalize("NFKC", text)

    text = text.lower()

    # minimal cleaning (DO NOT REMOVE diacritics)
    text = re.sub(r"[-/]", " ", text)

    # remove only excessive spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# =========================================================
# BUILD QUERY (ICD-LIKE WEIGHTING)
# =========================================================
def build_query(row):

    sanskrit = safe(row.get("namc_term_diacritical"))
    base_term = safe(row.get("namc_term"))
    dev = safe(row.get("namc_term_devanagari"))
    long_def = safe(row.get("long_definition"))
    short_def = safe(row.get("short_definition"))

    # weighted combination (same principle as ICD index_terms)
    combined = (
        (sanskrit + " ") * 2 +        # strongest signal
        (base_term + " ") * 1 +
        (dev + " ") * 1 +
        long_def + " " +
        short_def
    )

    combined = clean_text(combined)

    return combined

df["query"] = df.apply(build_query, axis=1)

# =========================================================
# FINAL DATASET
# =========================================================
df_final = df[[
    "namc_code",
    "namc_term_diacritical",
    "query"
]]

# =========================================================
# SAVE
# =========================================================
df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

# =========================================================
# VERIFY
# =========================================================
print("\n✅ Preprocessing completed")
print("💾 Saved:", OUTPUT_FILE)

print("\n🔍 Sample output:")
print(df_final.head(5).to_string(index=False))