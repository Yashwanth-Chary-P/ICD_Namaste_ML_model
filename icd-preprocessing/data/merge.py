import pandas as pd
import re
import ast
import unicodedata
import string

# =========================================================
# INPUT / OUTPUT
# =========================================================
INPUT_FILE = "icd_merged.csv"
OUTPUT_FILE = "icd_finalized.csv"

# =========================================================
# 1) LOAD MERGED ICD DATASET
# =========================================================
df = pd.read_csv(INPUT_FILE, encoding="utf-8")
df.columns = df.columns.str.lower().str.strip()

# =========================================================
# 2) RENAME COLUMNS TO CLEAN NAMES
# =========================================================
rename_map = {
    "fully specified name": "fsn",
    "description": "definition",
    "index terms": "icd_index_terms",
}
df = df.rename(columns=rename_map)

# =========================================================
# 3) ENSURE REQUIRED COLUMNS EXIST
# =========================================================
required_cols = [
    "code",
    "title",
    "fsn",
    "definition",
    "inclusions",
    "exclusions",
    "icd_index_terms",
    "chapter",
]

for col in required_cols:
    if col not in df.columns:
        df[col] = ""

# Keep only the columns we need, in a fixed order
df = df[required_cols]

# =========================================================
# 4) SAFE TEXT HELPERS
# =========================================================
def safe_text(x):
    """Convert NaN / None / numeric-like values safely to string."""
    if pd.isna(x):
        return ""
    text = str(x).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text

def parse_list_like_text(text):
    """
    Some cells may contain stringified lists like:
    "['term1', 'term2']"
    Convert them into:
    "term1 term2"
    """
    text = safe_text(text)
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                items = [safe_text(item) for item in parsed]
                return " ".join(item for item in items if item)
        except (ValueError, SyntaxError):
            pass
    return text

# =========================================================
# 5) CLEANING RULES
#    Important: we preserve Unicode letters/diacritics.
#    We do NOT strip Sanskrit characters like ā, ī, ū, ṣ, ṇ, ḥ.
# =========================================================
noise_patterns = [
    r"\bunspecified\b",
    r"\bother specified\b",
    r"\bnot elsewhere classified\b",
    r"\bnot elsewhere specified\b",
    r"\bnec\b",
    r"\bnos\b",
    r"\btm2\b",   # obvious dataset noise token
]

# Replace punctuation with spaces, but do not touch letters/diacritics.
punct_chars = set(string.punctuation)
punct_chars.update({
    "“", "”", "‘", "’", "—", "–", "…", "•", "·", "«", "»"
})
translator = str.maketrans({ch: " " for ch in punct_chars})

def clean_text(text):
    """
    Safe cleaning for healthcare NLP:
    1) parse list-like text
    2) normalize unicode
    3) lowercase
    4) replace punctuation with spaces
    5) remove noisy admin tokens
    6) collapse repeated spaces

    This keeps transliterated/Sanskrit words intact.
    """
    text = parse_list_like_text(text)

    # Unicode normalization keeps text consistent without removing diacritics
    text = unicodedata.normalize("NFKC", text)

    # Case normalization
    text = text.lower()

    # Convert punctuation/symbols into spaces
    text = text.translate(translator)

    # Remove common noise tokens
    for pattern in noise_patterns:
        text = re.sub(pattern, " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# =========================================================
# 6) CLEAN TEXT COLUMNS
# =========================================================
text_cols = [
    "title",
    "fsn",
    "definition",
    "inclusions",
    "exclusions",
    "icd_index_terms",
]

for col in text_cols:
    df[col] = df[col].apply(clean_text)

# Keep code and chapter as stable identifiers
df["code"] = df["code"].apply(safe_text).str.strip()
df["chapter"] = df["chapter"].apply(safe_text).str.strip()

# =========================================================
# 7) BUILD FINAL index_terms
#    Controlled duplication is intentional weighting.
#    We do NOT include exclusions in index_terms.
# =========================================================
def build_index_terms(row):
    title = safe_text(row["title"])
    fsn = safe_text(row["fsn"])
    icd_terms = safe_text(row["icd_index_terms"])
    inclusions = safe_text(row["inclusions"])
    definition = safe_text(row["definition"])

    combined = " ".join([
        " ".join([title] * 3),
        " ".join([fsn] * 2),
        " ".join([icd_terms] * 3),
        " ".join([inclusions] * 2),
        definition,
    ])

    combined = re.sub(r"\s+", " ", combined).strip()
    return combined

df["index_terms"] = df.apply(build_index_terms, axis=1)

# =========================================================
# 8) FINAL COLUMN ORDER
# =========================================================
final_order = [
    "code",
    "title",
    "fsn",
    "definition",
    "inclusions",
    "exclusions",
    "icd_index_terms",
    "chapter",
    "index_terms",
]
df = df[final_order]

# =========================================================
# 9) SAVE FINAL DATASET
# =========================================================
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

# =========================================================
# 10) VERIFY SAVED OUTPUT
# =========================================================
check = pd.read_csv(OUTPUT_FILE, encoding="utf-8")

print("✅ ICD preprocessing completed successfully.")
print(f"💾 Saved file: {OUTPUT_FILE}")
print("\nFinal columns:")
print(check.columns.tolist())
print("\nDataset shape:", check.shape)
print("\nSample output:")
print(check[["code", "title", "fsn", "index_terms"]].head(5).to_string(index=False))