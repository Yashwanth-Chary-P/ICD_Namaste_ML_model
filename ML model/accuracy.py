import pandas as pd
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# FILES
# ==============================
AYU_FILE = "ayurveda_final.csv"
EVAL_FILE = "eval.csv"
ICD_FILE = "tm2_final.csv"

# ==============================
# LOAD
# ==============================
ayu = pd.read_csv(AYU_FILE)
eval_df = pd.read_csv(EVAL_FILE)
icd = pd.read_csv(ICD_FILE)

# Normalize column names
ayu.columns = ayu.columns.str.lower().str.strip()
eval_df.columns = eval_df.columns.str.lower().str.strip()
icd.columns = icd.columns.str.lower().str.strip()

print("📥 Ayurveda:", len(ayu))
print("📥 Eval:", len(eval_df))
print("📥 ICD:", len(icd))

# ==============================
# CLEAN TEXT
# ==============================
def safe(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def clean(text):
    text = safe(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ==============================
# BUILD QUERY (AYURVEDA)
# ==============================
def build_query(row):
    return clean(
        (safe(row.get("namc_term_diacritical")) + " ") * 2 +
        safe(row.get("long_definition")) + " " +
        safe(row.get("short_definition"))
    )

ayu["query"] = ayu.apply(build_query, axis=1)
eval_df["query"] = eval_df.apply(build_query, axis=1)

# ==============================
# ICD TEXT (IMPORTANT FIX)
# ==============================
# We DO NOT use 'code' as text
# We use index_terms (rich semantic content)

icd["text"] = icd["index_terms"].fillna("").apply(clean)

# ==============================
# TF-IDF
# ==============================
vectorizer = TfidfVectorizer(max_features=5000)

all_text = list(ayu["query"]) + list(icd["text"])
vectorizer.fit(all_text)

eval_vec = vectorizer.transform(eval_df["query"])
icd_vec = vectorizer.transform(icd["text"])

print("✅ TF-IDF ready")

# ==============================
# SIMILARITY
# ==============================
sim = cosine_similarity(eval_vec, icd_vec)

# ==============================
# TOP-K
# ==============================
def top_k(arr, k=3):
    return arr.argsort()[-k:][::-1]

pred_top1 = []
pred_top3 = []

for i in range(len(eval_df)):
    idx = top_k(sim[i], 3)

    # IMPORTANT: using ICD "code" column
    pred_top1.append(icd.iloc[idx[0]]["code"])
    pred_top3.append(list(icd.iloc[idx]["code"]))

eval_df["pred_top1"] = pred_top1
eval_df["pred_top3"] = pred_top3

# ==============================
# EVALUATION
# ==============================
correct1 = 0
correct3 = 0

for i in range(len(eval_df)):
    true = safe(eval_df.iloc[i]["icd_code"])
    p1 = safe(eval_df.iloc[i]["pred_top1"])
    p3 = eval_df.iloc[i]["pred_top3"]

    # ⚠️ Normalize both (important)
    true = true.strip()
    p1 = p1.strip()

    if true == p1:
        correct1 += 1

    if true in p3:
        correct3 += 1

top1 = correct1 / len(eval_df)
top3 = correct3 / len(eval_df)

print("\n📊 RESULTS")
print("Top-1 Accuracy:", round(top1, 4))
print("Top-3 Accuracy:", round(top3, 4))

# ==============================
# SAVE
# ==============================
eval_df.to_csv("final_results.csv", index=False)

print("\n💾 Saved: final_results.csv")

# ==============================
# SAMPLE
# ==============================
print("\n🔍 Sample:")
print(eval_df[[
    "namc_code",
    "icd_code",
    "pred_top1",
    "pred_top3"
]].head(10))