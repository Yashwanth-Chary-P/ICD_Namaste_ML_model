import pandas as pd

# ==============================
# INPUT / OUTPUT
# ==============================
INPUT_FILE = "testing.csv"
OUTPUT_FILE = "eval.csv"

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(INPUT_FILE, encoding="utf-8")

# Clean column names
df.columns = df.columns.str.lower().str.strip()

print("📥 Loaded dataset")
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())

# ==============================
# RENAME ICD COLUMN
# ==============================
df.rename(columns={
    "primary index related thecode": "icd_code"
}, inplace=True)

# ==============================
# SELECT REQUIRED COLUMNS
# ==============================
required_columns = [
    "namc_code",
    "namc_term_diacritical",
    "short_definition",
    "long_definition",
    "icd_code"
]

df_clean = df[required_columns].copy()

# ==============================
# OPTIONAL CLEANING (SAFE)
# ==============================
df_clean.fillna("", inplace=True)

# ==============================
# SAVE OUTPUT
# ==============================
df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

# ==============================
# VERIFY
# ==============================
print("\n✅ Clean mapping dataset created")
print("💾 Saved as:", OUTPUT_FILE)
print("📊 Total rows:", len(df_clean))
print("📊 Unique ICD codes:", df_clean["icd_code"].nunique())

print("\n🔍 Sample:")
print(df_clean.head(5))