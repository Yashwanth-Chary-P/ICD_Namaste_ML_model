import pandas as pd

# ==============================
# INPUT / OUTPUT
# ==============================
INPUT_FILE = "AYURVEDA.csv"   # your raw file
OUTPUT_FILE = "ayurveda_final.csv"

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(INPUT_FILE, encoding="utf-8")

# Normalize column names
df.columns = df.columns.str.lower().str.strip()

print("📥 Loaded dataset")
print("Columns found:", df.columns.tolist())
print("Total rows:", len(df))

# ==============================
# RENAME IMPORTANT COLUMNS
# ==============================
df.rename(columns={
    "primary index related thecode": "icd_code"
}, inplace=True)

# ==============================
# ENSURE REQUIRED COLUMNS EXIST
# ==============================
required_columns = {
    "namc_code": "",
    "namc_term_diacritical": "",
    "short_definition": "",
    "long_definition": "",
    "icd_code": ""
}

for col in required_columns:
    if col not in df.columns:
        print(f"⚠️ Missing column: {col} → creating empty")
        df[col] = ""

# ==============================
# SELECT ONLY REQUIRED COLUMNS
# ==============================
df_final = df[list(required_columns.keys())].copy()

# ==============================
# CLEAN TEXT (SAFE)
# ==============================
df_final.fillna("", inplace=True)

# Strip spaces
for col in df_final.columns:
    df_final[col] = df_final[col].astype(str).str.strip()

# ==============================
# REMOVE EMPTY ICD ROWS (OPTIONAL)
# ==============================
# Uncomment if needed:
# df_final = df_final[df_final["icd_code"] != ""]

# ==============================
# SAVE FINAL DATASET
# ==============================
df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

# ==============================
# VERIFY
# ==============================
print("\n✅ Dataset rebuilt successfully")
print("💾 Saved as:", OUTPUT_FILE)
print("📊 Total rows:", len(df_final))
print("📊 Unique ICD codes:", df_final["icd_code"].nunique())

print("\n🔍 Sample:")
print(df_final.head(5))