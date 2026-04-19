import pandas as pd
import re

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("AYURVEDA.csv", encoding="utf-8")

# Normalize column names
df.columns = df.columns.str.lower().str.strip()

print("📥 Loaded rows:", len(df))


# ==============================
# EXTRACT TM2 CODE (STRICT)
# ==============================
def extract_tm2(namc_code):
    if pd.isna(namc_code):
        return ""

    text = str(namc_code).replace("\xa0", " ").strip()

    # Case 1: TM2 at beginning (SR11, SQ00, SP9Y)
    match = re.match(r"^(S[A-Z][0-9A-Z]+)", text)
    if match:
        return match.group(1)

    # Case 2: TM2 inside brackets
    match = re.search(r"\((S[A-Z][0-9A-Z]+)\)", text)
    if match:
        return match.group(1)

    return ""


df["tm2_code"] = df["namc_code"].apply(extract_tm2)


# ==============================
# CLEAN NAMC CODE (SAFE)
# ==============================
def clean_namc(namc_code):
    if pd.isna(namc_code):
        return ""

    text = str(namc_code)

    # Fix unicode spacing
    text = text.replace("\xa0", " ")

    # Remove ONLY TM2 at start (SR11, SQ00)
    text = re.sub(r"^S[A-Z][0-9A-Z]+\s*", "", text)

    # Remove ONLY TM2 inside brackets
    text = re.sub(r"\(S[A-Z][0-9A-Z]+\)", "", text)

    # Remove brackets but KEEP content
    text = re.sub(r"[()]", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["namc_code_clean"] = df["namc_code"].apply(clean_namc)


# ==============================
# CLEAN TEXT (PRESERVE SANSKRIT)
# ==============================
def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Only fix spacing — DO NOT lowercase or normalize accents
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["namc_term_diacritical"] = df["namc_term_diacritical"].apply(clean_text)
df["short_definition"] = df["short_definition"].apply(clean_text)
df["long_definition"] = df["long_definition"].apply(clean_text)


# ==============================
# DATASET 1: PURE NAMASTE
# ==============================
ayu_df = df[[
    "namc_code_clean",
    "namc_term_diacritical",
    "short_definition",
    "long_definition"
]].copy()

ayu_df.rename(columns={
    "namc_code_clean": "namc_code"
}, inplace=True)

# Remove completely empty codes
ayu_df = ayu_df[ayu_df["namc_code"] != ""]

ayu_df.to_csv("ayurveda_only.csv", index=False)

print("\n✅ Pure NAMASTE dataset created")
print("Rows:", len(ayu_df))


# ==============================
# DATASET 2: EVALUATION DATASET
# ==============================
tm2_df = df[df["tm2_code"] != ""].copy()

# ICD column (correct one)
tm2_df["icd_code"] = tm2_df["primary index related"]

# Remove invalid ICD rows
tm2_df = tm2_df[
    (tm2_df["icd_code"].notna()) &
    (tm2_df["icd_code"] != "-") &
    (tm2_df["icd_code"] != "")
]

tm2_df = tm2_df[[
    "namc_code_clean",
    "tm2_code",
    "icd_code",
    "namc_term_diacritical",
    "short_definition",
    "long_definition"
]]

tm2_df.rename(columns={
    "namc_code_clean": "namc_code"
}, inplace=True)

tm2_df.to_csv("tm2_eval.csv", index=False)

print("\n✅ Evaluation dataset created")
print("Rows:", len(tm2_df))


# ==============================
# VALIDATION CHECKS
# ==============================
print("\n📊 SUMMARY")
print("Total rows:", len(df))
print("NAMASTE dataset:", len(ayu_df))
print("Evaluation dataset:", len(tm2_df))

print("\n🔍 SAMPLE CHECK (IMPORTANT)")
for i in range(10):
    original = df["namc_code"].iloc[i]
    cleaned = df["namc_code_clean"].iloc[i]
    tm2 = df["tm2_code"].iloc[i]
    print(f"{original}  →  {cleaned}  | TM2: {tm2}")