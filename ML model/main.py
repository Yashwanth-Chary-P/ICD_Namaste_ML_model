import pandas as pd
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# FILE PATHS
# ==============================
AYURVEDA_FILE = "ayurveda_final.csv"
ICD_FILE = "tm2_final.csv"

# ==============================
# LOAD DATA
# ==============================
ayu = pd.read_csv(AYURVEDA_FILE, encoding="utf-8")
icd = pd.read_csv(ICD_FILE, encoding="utf-8")

# Normalize column names
ayu.columns = ayu.columns.str.lower().str.strip()
icd.columns = icd.columns.str.lower().str.strip()

print("📥 Ayurveda rows:", len(ayu))
print("📥 ICD rows:", len(icd))

# ==============================
# SAFE TEXT
# ==============================
def safe(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

# ==============================
# CLEAN TEXT
# ==============================
def clean_text(text):
    text = safe(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==============================
# BUILD QUERY (ROBUST)
# ==============================
def build_query(row):
    sanskrit = safe(row.get("namc_term_diacritical", ""))
    short_def = safe(row.get("short_definition", ""))
    long_def = safe(row.get("long_definition", ""))

    return clean_text(
        (sanskrit + " ") * 2 +
        long_def + " " +
        short_def
    )

ayu["query"] = ayu.apply(build_query, axis=1)

# ==============================
# PREPARE ICD TEXT
# ==============================
if "index_terms" in icd.columns:
    icd["text"] = icd["index_terms"].fillna("")
else:
    icd["text"] = (
        icd.get("title", "").astype(str) + " " +
        icd.get("definition", "").astype(str)
    )

icd["text"] = icd["text"].apply(clean_text)

# ==============================
# TF-IDF
# ==============================
vectorizer = TfidfVectorizer(max_features=5000)

# Fit on combined corpus
all_text = list(ayu["query"]) + list(icd["text"])
vectorizer.fit(all_text)

# Transform
ayu_vec = vectorizer.transform(ayu["query"])
icd_vec = vectorizer.transform(icd["text"])

print("✅ TF-IDF built")

# ==============================
# SIMILARITY
# ==============================
sim_matrix = cosine_similarity(ayu_vec, icd_vec)

# ==============================
# TOP-K PREDICTION
# ==============================
def get_top_k(sim_row, k=3):
    return sim_row.argsort()[-k:][::-1]

top1 = []
top3 = []

for i in range(len(ayu)):
    idx = get_top_k(sim_matrix[i], k=3)

    top1.append(icd.iloc[idx[0]]["code"])
    top3.append(list(icd.iloc[idx]["code"]))

ayu["pred_top1"] = top1
ayu["pred_top3"] = top3

# ==============================
# EVALUATION (ONLY IF LABEL EXISTS)
# ==============================
if "icd_code" in ayu.columns:
    correct1 = 0
    correct3 = 0
    total = 0

    for i in range(len(ayu)):
        true = safe(ayu.iloc[i]["icd_code"])
        if true == "":
            continue

        total += 1

        if true == safe(ayu.iloc[i]["pred_top1"]):
            correct1 += 1

        if true in ayu.iloc[i]["pred_top3"]:
            correct3 += 1

    if total > 0:
        print("\n📊 Evaluation")
        print("Top-1 Accuracy:", round(correct1 / total, 4))
        print("Top-3 Accuracy:", round(correct3 / total, 4))
    else:
        print("\n⚠️ No labeled data for evaluation")

# ==============================
# SAVE OUTPUT
# ==============================
OUTPUT_FILE = "results.csv"
ayu.to_csv(OUTPUT_FILE, index=False)

print("\n💾 Results saved:", OUTPUT_FILE)

# ==============================
# SAMPLE OUTPUT
# ==============================
print("\n🔍 Sample:")
print(ayu[[
    "namc_code",
    "icd_code",
    "pred_top1",
    "pred_top3"
]].head(10))