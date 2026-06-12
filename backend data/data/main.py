import pandas as pd

# -----------------------------
# FILES
# -----------------------------
SIDDHA_FILE = "siddha_processed.csv"
UNANI_FILE = "unani_processed.csv"
OUTPUT_FILE = "traditional_combined_full.csv"

# -----------------------------
# LOAD
# -----------------------------
sid = pd.read_csv(SIDDHA_FILE)
uni = pd.read_csv(UNANI_FILE)

# Clean column names a little: strip spaces
sid.columns = [str(c).strip() for c in sid.columns]
uni.columns = [str(c).strip() for c in uni.columns]

# -----------------------------
# ADD STANDARD FIELDS
# Keep original columns intact
# -----------------------------
sid["system"] = "siddha"
sid["code"] = sid.get("NAMC_CODE", "")
sid["term"] = sid.get("NAMC_TERM", "")
sid["query"] = sid.get("query", "")

uni["system"] = "unani"
uni["code"] = uni.get("NUMC_CODE", "")
uni["term"] = uni.get("NUMC_TERM", "")
uni["query"] = uni.get("query", "")

# Optional: a unified display label for UI/backend convenience
sid["display_name"] = sid["term"]
uni["display_name"] = uni["term"]

# -----------------------------
# MAKE COLUMN SET UNION
# Preserve all columns from both datasets
# -----------------------------
all_columns = list(sid.columns)
for col in uni.columns:
    if col not in all_columns:
        all_columns.append(col)

# Add missing columns to each dataframe
for col in all_columns:
    if col not in sid.columns:
        sid[col] = ""
    if col not in uni.columns:
        uni[col] = ""

# Reorder to same column order
sid = sid[all_columns]
uni = uni[all_columns]

# -----------------------------
# MERGE
# -----------------------------
combined = pd.concat([sid, uni], ignore_index=True)

# -----------------------------
# OPTIONAL CLEANUP
# - fill NaN with empty string
# - keep everything as display-safe text
# -----------------------------
combined = combined.fillna("")

# -----------------------------
# SAVE
# -----------------------------
combined.to_csv(OUTPUT_FILE, index=False)

print("Merged successfully!")
print("Rows:", len(combined))
print("Columns:", list(combined.columns))
print(combined.head(3))