import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ==============================
# LOAD
# ==============================
ayu = pd.read_csv("ayurveda_with_tm2_clean.csv")
eval_df = pd.read_csv("eval_dataset_final.csv")
tm2 = pd.read_csv("tm2.csv")

print("📥 Loaded:")
print("Ayurveda:", len(ayu))
print("Eval:", len(eval_df))
print("TM2:", len(tm2))

# ==============================
# CLEAN
# ==============================
ayu.columns = ayu.columns.str.strip().str.lower()
eval_df.columns = eval_df.columns.str.strip().str.lower()
tm2.columns = tm2.columns.str.strip().str.lower()

# rename tm2 columns
tm2 = tm2.rename(columns={
    "code": "code",
    "index terms": "index_terms"
})

# fill
for col in ["namc_term_diacritical", "short_definition", "long_definition"]:
    ayu[col] = ayu.get(col, "").fillna("").astype(str)

tm2["title"] = tm2.get("title", "").fillna("").astype(str)
tm2["index_terms"] = tm2.get("index_terms", "").fillna("").astype(str)

# ==============================
# BUILD QUERY
# ==============================
ayu["final_query"] = (
    (ayu["namc_term_diacritical"] + " ") * 4 +
    ayu["short_definition"] + " " +
    ayu["long_definition"]
)

# ==============================
# BUILD TM2 TEXT
# ==============================
tm2["final_text"] = (
    (tm2["title"] + " ") * 3 +
    (tm2["index_terms"] + " ") * 4
)

# ==============================
# TF-IDF
# ==============================
start = time.time()

vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))

tfidf_tm2 = vectorizer.fit_transform(tm2["final_text"])
tfidf_ayu = vectorizer.transform(ayu["final_query"])

similarity = cosine_similarity(tfidf_ayu, tfidf_tm2)

print("⏱️ TF-IDF time:", round(time.time() - start, 2))

# ==============================
# MAPPING
# ==============================
results = []

for i in tqdm(range(len(ayu)), desc="Mapping"):
    sims = similarity[i]
    top_indices = np.argsort(sims)[-3:][::-1]

    preds = [tm2.iloc[j]["code"] for j in top_indices]

    results.append({
        "namc_code": ayu.iloc[i]["namc_code"],
        "top1_pred": preds[0],
        "top2_pred": preds[1],
        "top3_pred": preds[2],
    })

mapping_df = pd.DataFrame(results)

# ==============================
# EVALUATION
# ==============================
merged = pd.merge(eval_df, mapping_df, on="namc_code")

top1 = sum(merged["tm2_code"] == merged["top1_pred"])
top3 = sum(merged.apply(lambda r: r["tm2_code"] in [r["top1_pred"], r["top2_pred"], r["top3_pred"]], axis=1))

print("\n📊 RESULTS (NO DICTIONARY)")
print("Top-1:", round(top1 / len(merged), 4))
print("Top-3:", round(top3 / len(merged), 4))