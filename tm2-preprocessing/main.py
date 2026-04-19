import pandas as pd
import re
import ast
import unicodedata
import string

# =========================================================
# INPUT / OUTPUT
# =========================================================
INPUT_FILE = "tm2.csv"
OUTPUT_FILE = "tm2_final.csv"

# =========================================================
# 1) LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_FILE, encoding="utf-8")
df.columns = df.columns.str.lower().str.strip()

print("📥 Loaded ICD dataset")
print("Columns:", df.columns.tolist())

# =========================================================
# 2) RENAME COLUMNS
# =========================================================
df.rename(columns={
    "fully specified name": "fsn",
    "description": "definition",
    "index terms": "icd_index_terms"
}, inplace=True)

# =========================================================
# 3) ENSURE REQUIRED COLUMNS
# =========================================================
required_cols = [
    "code",
    "title",
    "fsn",
    "definition",
    "inclusions",
    "exclusions",
    "icd_index_terms",
    "chapter"
]

for col in required_cols:
    if col not in df.columns:
        df[col] = ""

df = df[required_cols]

# =========================================================
# 4) SAFE TEXT
# =========================================================
def safe(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

# =========================================================
# 5) PARSE LIST-LIKE TEXT
# =========================================================
def parse_list(text):
    text = safe(text)
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return " ".join(str(x) for x in parsed if str(x).strip())
        except:
            pass
    return text

# =========================================================
# 6) CLEAN TEXT (SAFE FOR MEDICAL + SANSKRIT)
# =========================================================
noise_patterns = [
    r"\bunspecified\b",
    r"\bother specified\b",
    r"\bnot elsewhere classified\b",
    r"\bnot elsewhere specified\b",
    r"\bnec\b",
    r"\bnos\b"
]

punct_chars = set(string.punctuation)
translator = str.maketrans({ch: " " for ch in punct_chars})

def clean_text(text):
    text = parse_list(text)

    # Unicode normalize (keep diacritics)
    text = unicodedata.normalize("NFKC", text)

    text = text.lower()

    # Replace punctuation
    text = text.translate(translator)

    # Remove noise words
    for pattern in noise_patterns:
        text = re.sub(pattern, " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# =========================================================
# 7) APPLY CLEANING
# =========================================================
text_cols = [
    "title",
    "fsn",
    "definition",
    "inclusions",
    "exclusions",
    "icd_index_terms"
]

for col in text_cols:
    df[col] = df[col].apply(clean_text)

# =========================================================
# 8) BUILD INDEX_TERMS (CORE STEP)
# =========================================================
def build_index_terms(row):

    title = safe(row["title"])
    fsn = safe(row["fsn"])
    icd_terms = safe(row["icd_index_terms"])
    inclusions = safe(row["inclusions"])
    definition = safe(row["definition"])

    combined = " ".join([
        " ".join([title] * 3),
        " ".join([fsn] * 2),
        " ".join([icd_terms] * 3),
        " ".join([inclusions] * 2),
        definition
    ])

    combined = re.sub(r"\s+", " ", combined).strip()
    return combined

df["index_terms"] = df.apply(build_index_terms, axis=1)

# =========================================================
# 9) FINAL SAVE
# =========================================================
final_cols = [
    "code",
    "title",
    "fsn",
    "definition",
    "inclusions",
    "exclusions",
    "icd_index_terms",
    "chapter",
    "index_terms"
]

df = df[final_cols]

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

# =========================================================
# 10) VERIFY
# =========================================================
print("\n✅ ICD preprocessing completed")
print("💾 Saved:", OUTPUT_FILE)

print("\n📊 Shape:", df.shape)

print("\n🔍 Sample:")
print(df[["code", "title", "index_terms"]].head(5).to_string(index=False))