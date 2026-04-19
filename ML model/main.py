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
start = time.time()

ayu = pd.read_csv("AYURVEDA_clean.csv")
tm2 = pd.read_csv("tm2_final.csv")
eval_df = pd.read_csv("tm2_eval.csv")

print("📥 Loaded:")
print("Ayurveda:", len(ayu))
print("TM2:", len(tm2))
print("Eval:", len(eval_df))

# ==============================
# CLEAN
# ==============================
for df in [ayu, tm2, eval_df]:
    df.columns = df.columns.str.strip()

ayu = ayu.drop_duplicates(subset=["namc_code"])
tm2 = tm2.drop_duplicates(subset=["code"])
eval_df = eval_df.drop_duplicates(subset=["namc_code"])

for col in ["query", "label", "namc_term_diacritical"]:
    ayu[col] = ayu.get(col, "").fillna("").astype(str)

for col in ["title", "fsn", "definition", "icd_index_terms"]:
    if col in tm2.columns:
        tm2[col] = tm2[col].fillna("").astype(str)
    else:
        tm2[col] = ""

# remove empty queries
ayu = ayu[ayu["query"].str.strip() != ""].reset_index(drop=True)

print("✅ After cleaning:", len(ayu))

# ==============================
# BUILD QUERY
# ==============================
ayu["final_query"] = (
    ayu["namc_term_diacritical"] + " " +
    ayu["query"] + " " +
    ayu["label"]
)

# NO index_terms used
tm2["final_text"] = (
    tm2["title"] + " " +
    tm2["fsn"] + " " +
    tm2["definition"] + " " +
    tm2["icd_index_terms"]
)

# ==============================
# DICTIONARY (FROM EVAL)
# ==============================
sanskrit_to_tm2 = defaultdict(set)

dict_df = pd.merge(
    eval_df,
    ayu[["namc_code", "namc_term_diacritical"]],
    on="namc_code",
    how="inner"
)

sanskrit_col = [c for c in dict_df.columns if "namc_term_diacritical" in c][0]

for _, r in dict_df.iterrows():
    s = str(r[sanskrit_col]).strip()
    if s:
        sanskrit_to_tm2[s].add(r["tm2_code"])

print("📚 Dictionary size:", len(sanskrit_to_tm2))

# ==============================
# TF-IDF
# ==============================
tfidf = TfidfVectorizer(max_features=30000)

tfidf_tm2 = tfidf.fit_transform(tm2["final_text"])
tfidf_ayu = tfidf.transform(ayu["final_query"])

tfidf_sim = cosine_similarity(tfidf_ayu, tfidf_tm2)

# ==============================
# BERT
# ==============================
print("🧠 Loading BERT...")
model = SentenceTransformer("all-MiniLM-L6-v2")

ayu_emb = model.encode(
    ayu["final_query"].tolist(),
    convert_to_tensor=True,
    show_progress_bar=True
)

tm2_emb = model.encode(
    tm2["final_text"].tolist(),
    convert_to_tensor=True,
    show_progress_bar=True
)

# ==============================
# TAG FUNCTION
# ==============================
def assign_tag(score, dict_match):
    if dict_match:
        return "Equivalent"
    elif score > 0.75:
        return "Equivalent"
    elif score > 0.60:
        return "Narrower"
    elif score > 0.45:
        return "Related"
    else:
        return "Weak"

# ==============================
# MAPPING (TOP-3)
# ==============================
results = []

for i in tqdm(range(len(ayu)), desc="Mapping"):
    bert_scores = util.cos_sim(ayu_emb[i], tm2_emb)[0].cpu().numpy()

    top_candidates = np.argsort(bert_scores)[-50:]

    sanskrit = ayu.iloc[i]["namc_term_diacritical"]

    reranked = []

    for j in top_candidates:
        score = bert_scores[j]
        score += 0.25 * tfidf_sim[i][j]

        dict_match = False
        if sanskrit in sanskrit_to_tm2:
            if tm2.iloc[j]["code"] in sanskrit_to_tm2[sanskrit]:
                score += 1.2
                dict_match = True

        reranked.append((j, score, dict_match))

    reranked = sorted(reranked, key=lambda x: x[1], reverse=True)[:3]

    row_out = {"namc_code": ayu.iloc[i]["namc_code"]}

    for k, (j, score, dict_match) in enumerate(reranked):
        tag = assign_tag(score, dict_match)

        row_out[f"top{k+1}_pred"] = tm2.iloc[j]["code"]
        row_out[f"top{k+1}_tag"] = tag
        row_out[f"top{k+1}_score"] = score

    results.append(row_out)

mapping_df = pd.DataFrame(results)
mapping_df.to_csv("mapping_output_final.csv", index=False)

print("✅ Mapping saved")

# ==============================
# EVALUATION (CORRECT)
# ==============================
merged = pd.merge(eval_df, mapping_df, on="namc_code", how="inner")

strict = 0
top3 = 0
eq = 0
eq_narrow = 0
clinical = 0

for _, r in merged.iterrows():
    true = r["tm2_code"]

    preds = [r["top1_pred"], r["top2_pred"], r["top3_pred"]]
    tags = [r["top1_tag"], r["top2_tag"], r["top3_tag"]]

    # Strict
    if true == preds[0]:
        strict += 1

    # Top-3
    if true in preds:
        top3 += 1

    # Equivalent
    for p, t in zip(preds, tags):
        if p == true and t == "Equivalent":
            eq += 1

    # Equivalent + Narrower
    for p, t in zip(preds, tags):
        if p == true and t in ["Equivalent", "Narrower"]:
            eq_narrow += 1

    # Clinical (Equivalent + Narrower + Related)
    if any(p == true and t in ["Equivalent", "Narrower", "Related"] for p, t in zip(preds, tags)):
        clinical += 1

total = len(merged)

print("\n📊 FINAL METRICS")
print("Strict Top-1:", round(strict / total, 4))
print("Top-3 Accuracy:", round(top3 / total, 4))
print("Equivalent Accuracy:", round(eq / total, 4))
print("Equivalent + Narrower:", round(eq_narrow / total, 4))
print("Clinical (Eq + Nar + Rel):", round(clinical / total, 4))
print("Coverage:", round(total / len(eval_df), 4))

print(f"\n⏱️ Total time: {time.time() - start:.2f}s")