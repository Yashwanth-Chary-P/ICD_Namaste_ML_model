import pandas as pd
import numpy as np
import torch
import time
from tqdm import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# ==============================
# LOAD
# ==============================
start_time = time.time()

ayu = pd.read_csv("AYURVEDA_clean.csv")
tm2 = pd.read_csv("tm2_final.csv")
eval_df = pd.read_csv("tm2_eval.csv")

print("📥 Loaded")

# ==============================
# CLEAN
# ==============================
ayu.columns = ayu.columns.str.strip()
tm2.columns = tm2.columns.str.strip()
eval_df.columns = eval_df.columns.str.strip()

ayu = ayu.drop_duplicates(subset=["namc_code"])
tm2 = tm2.drop_duplicates(subset=["code"])
eval_df = eval_df.drop_duplicates(subset=["namc_code"])

# fill safely
for col in ["query", "label", "namc_term_diacritical"]:
    if col in ayu.columns:
        ayu[col] = ayu[col].fillna("").astype(str)
    else:
        ayu[col] = ""

tm2["index_terms"] = tm2["index_terms"].fillna("").astype(str)

# remove empty
ayu = ayu[ayu["query"].str.strip() != ""].reset_index(drop=True)

# ==============================
# QUERY
# ==============================
ayu["final_query"] = (
    ayu["namc_term_diacritical"] + " " +
    ayu["query"] + " " +
    ayu["label"]
)

tm2["final_text"] = tm2["index_terms"]

# ==============================
# DICTIONARY (SAFE)
# ==============================
sanskrit_to_tm2 = defaultdict(set)

dict_df = pd.merge(
    eval_df,
    ayu[["namc_code", "namc_term_diacritical"]],
    on="namc_code",
    how="inner"
)

# 🔥 detect correct column automatically
sanskrit_col = None
for c in dict_df.columns:
    if "namc_term_diacritical" in c:
        sanskrit_col = c
        break

if sanskrit_col is None:
    raise ValueError("❌ Sanskrit column not found")

print("✅ Using Sanskrit column:", sanskrit_col)

for _, r in dict_df.iterrows():
    s = str(r[sanskrit_col]).strip()
    if s:
        sanskrit_to_tm2[s].add(r["tm2_code"])

print("📚 Dictionary size:", len(sanskrit_to_tm2))

# ==============================
# TF-IDF
# ==============================
tfidf = TfidfVectorizer(max_features=20000)

tfidf_tm2 = tfidf.fit_transform(tm2["final_text"])
tfidf_ayu = tfidf.transform(ayu["final_query"])

tfidf_sim = cosine_similarity(tfidf_ayu, tfidf_tm2)

# ==============================
# BERT
# ==============================
print("🧠 Loading BERT...")
model = SentenceTransformer("all-MiniLM-L6-v2")

ayu_emb = model.encode(ayu["final_query"].tolist(), convert_to_tensor=True, show_progress_bar=True)
tm2_emb = model.encode(tm2["final_text"].tolist(), convert_to_tensor=True, show_progress_bar=True)

# ==============================
# HYBRID SCORING
# ==============================
results = []

for i in tqdm(range(len(ayu)), desc="Hybrid Mapping"):
    tfidf_scores = tfidf_sim[i]
    bert_scores = util.cos_sim(ayu_emb[i], tm2_emb)[0].cpu().numpy()

    sanskrit = ayu.iloc[i]["namc_term_diacritical"]

    combined = []

    for j in range(len(tm2)):
        score = (
            0.4 * tfidf_scores[j] +
            0.6 * bert_scores[j]
        )

        # dictionary boost
        if sanskrit in sanskrit_to_tm2:
            if tm2.iloc[j]["code"] in sanskrit_to_tm2[sanskrit]:
                score += 0.5

        combined.append(score)

    combined = np.array(combined)

    top_idx = combined.argsort()[-3:][::-1]

    preds = tm2.iloc[top_idx]["code"].tolist()

    results.append({
        "namc_code": ayu.iloc[i]["namc_code"],
        "top1_pred": preds[0],
        "top2_pred": preds[1],
        "top3_pred": preds[2],
    })

mapping_df = pd.DataFrame(results)
mapping_df.to_csv("mapping_output_hybrid.csv", index=False)

print("✅ Mapping saved")

# ==============================
# EVALUATION
# ==============================
merged = pd.merge(eval_df, mapping_df, on="namc_code", how="inner")

top1 = sum(merged["tm2_code"] == merged["top1_pred"])
top3 = sum(merged.apply(lambda r: r["tm2_code"] in [r["top1_pred"], r["top2_pred"], r["top3_pred"]], axis=1))

total = len(merged)

print("\n📊 HYBRID RESULTS")
print("Top-1:", round(top1 / total, 4))
print("Top-3:", round(top3 / total, 4))
print("Coverage:", round(total / len(eval_df), 4))

print(f"\n⏱️ Total time: {time.time() - start_time:.2f}s")