import pandas as pd
import unicodedata

# =========================================================
# CONFIG
# =========================================================
INPUT_FILE = "UNANI.csv"
OUTPUT_FILE = "unani_processed.csv"

# Columns
COL_ARABIC = "Arabic_term"
COL_TERM = "NUMC_TERM"
COL_SHORT = "Short_definition"
COL_LONG = "Long_definition"

# =========================================================
# SAFE GET
# =========================================================
def safe_get(row, col):
    if col in row and pd.notna(row[col]) and row[col] != "":
        return str(row[col])
    return ""

# =========================================================
# NORMALIZATION
# =========================================================
def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = " ".join(text.split())
    return text

# =========================================================
# BUILD QUERY
# =========================================================
def build_query(row):
    arabic = safe_get(row, COL_ARABIC)
    term = safe_get(row, COL_TERM)
    short = safe_get(row, COL_SHORT)
    long_def = safe_get(row, COL_LONG)

    combined = (
        (term + " ") * 4 +
        (arabic + " ") * 4 +
        (short + " ") * 3 +
        (long_def + " ")
    )

    return normalize_text(combined)

# =========================================================
# FILTER INVALID ROWS (IMPORTANT)
# =========================================================
def is_valid(row):
    term = safe_get(row, COL_TERM)
    arabic = safe_get(row, COL_ARABIC)
    short = safe_get(row, COL_SHORT)

    # Remove rows like:
    # UM, UM-DIS, empty entries
    if term == "" and arabic == "" and short == "":
        return False
    return True

# =========================================================
# MAIN
# =========================================================
def main():
    print("📥 Loading Unani dataset...")
    df = pd.read_csv(INPUT_FILE)

    print(f"✅ Rows loaded: {len(df)}")

    # Ensure columns exist
    for col in [COL_ARABIC, COL_TERM, COL_SHORT, COL_LONG]:
        if col not in df.columns:
            print(f"⚠ Missing column: {col} → creating empty")
            df[col] = ""

    print("🧹 Filtering invalid rows...")
    df = df[df.apply(is_valid, axis=1)]

    print(f"✅ Rows after cleaning: {len(df)}")

    print("⚙ Building query column...")
    df["query"] = df.apply(build_query, axis=1)

    # Optional debug
    df["normalized_term"] = df[COL_TERM].apply(
        lambda x: normalize_text(str(x))
    )

    print("💾 Saving processed dataset...")
    df.to_csv(OUTPUT_FILE, index=False)

    print("🚀 Done! File saved:", OUTPUT_FILE)

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    main()