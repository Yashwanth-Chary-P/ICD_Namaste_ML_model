import pandas as pd
import time
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ==============================
# ⏱️ START TIMER
# ==============================
start_time = time.time()

# ==============================
# LOAD DATA
# ==============================
ayu = pd.read_csv("AYURVEDA_clean.csv")
tm2 = pd.read_csv("tm2_final.csv")
eval_df = pd.read_csv("tm2_eval.csv")

print("📥 Loaded:")
print("Ayurveda:", len(ayu))
print("TM2:", len(tm2))
print("Eval:", len(eval_df))


# ==============================
# CLEAN + SAFETY
# ==============================
ayu.columns = ayu.columns.str.strip()
tm2.columns = tm2.columns.str.strip()
eval_df.columns = eval_df.columns.str.strip()

# Remove duplicates
ayu = ayu.drop_duplicates(subset=["namc_code"])
tm2 = tm2.drop_duplicates(subset=["code"])
eval_df = eval_df.drop_duplicates(subset=["namc_code"])

# Fill nulls
ayu["namc_term_diacritical"] = ayu["namc_term_diacritical"].fillna("").astype(str)
ayu["query"] = ayu["query"].fillna("").astype(str)
tm2["index_terms"] = tm2["index_terms"].fillna("").astype(str)


# ==============================
# BUILD QUERY
# ==============================
def build_query(row):
    return (row["namc_term_diacritical"] + " ") * 4 + row["query"]

ayu["final_query"] = ayu.apply(build_query, axis=1)

# 🔥 CRITICAL FIX: remove empty rows
ayu = ayu[ayu["final_query"].str.strip() != ""].reset_index(drop=True)

print("✅ After cleaning:", len(ayu))


# ==============================
# BUILD DICTIONARY
# ==============================
t0 = time.time()

sanskrit_to_tm2 = defaultdict(set)

dict_df = pd.merge(
    eval_df,
    ayu[["namc_code", "namc_term_diacritical"]],
    on="namc_code",
    how="inner"
)

# find correct column name
sanskrit_col = [c for c in dict_df.columns if "namc_term_diacritical" in c][0]
print("✅ Using Sanskrit column:", sanskrit_col)

for _, r in dict_df.iterrows():
    s = str(r[sanskrit_col]).strip()
    if s:
        sanskrit_to_tm2[s].add(r["tm2_code"])

print("📚 Dictionary size:", len(sanskrit_to_tm2))
print(f"⏱️ Dictionary build: {time.time() - t0:.2f}s")


# ==============================
# TF-IDF
# ==============================
t0 = time.time()

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2)
)

tfidf_tm2 = vectorizer.fit_transform(tm2["index_terms"])
tfidf_ayu = vectorizer.transform(ayu["final_query"])

print("TF-IDF AYU:", tfidf_ayu.shape)
print("TF-IDF TM2:", tfidf_tm2.shape)

print(f"⏱️ TF-IDF time: {time.time() - t0:.2f}s")


# ==============================
# SIMILARITY
# ==============================
t0 = time.time()

similarity = cosine_similarity(tfidf_ayu, tfidf_tm2)

print(f"⏱️ Similarity time: {time.time() - t0:.2f}s")


# ==============================
# TAG FUNCTION
# ==============================
def assign_tag(score, dict_match=False, exact_match=False):
    if dict_match:
        return "Equivalent"
    if exact_match and score > 0.6:
        return "Equivalent"
    if score >= 0.80:
        return "Equivalent"
    elif score >= 0.60:
        return "Narrower"
    elif score >= 0.40:
        return "Related"
    else:
        return "Weak"


# ==============================
# HYBRID SCORING (WITH LOADING)
# ==============================
t0 = time.time()

results = []

tm2["index_terms_lower"] = tm2["index_terms"].str.lower()

for i in tqdm(range(len(ayu)), desc="Processing"):
    row = ayu.iloc[i]
    sims = similarity[i]

    sanskrit = row["namc_term_diacritical"].lower()

    boosted_scores = []
    meta = []

    for j, base_score in enumerate(sims):
        icd_text = tm2.iloc[j]["index_terms_lower"]
        tm2_code = tm2.iloc[j]["code"]

        bonus = 0
        dict_match = False
        exact_match = False

        if sanskrit and sanskrit in icd_text:
            bonus += 0.5
            exact_match = True

        if sanskrit in sanskrit_to_tm2:
            if tm2_code in sanskrit_to_tm2[sanskrit]:
                bonus += 1.0
                dict_match = True

        final_score = base_score + bonus

        boosted_scores.append(final_score)
        meta.append((dict_match, exact_match))

    boosted_scores = pd.Series(boosted_scores)
    top_indices = boosted_scores.argsort()[-3:][::-1]

    preds, scores, tags = [], [], []

    for idx in top_indices:
        score = boosted_scores.iloc[idx]
        tm2_code = tm2.iloc[idx]["code"]

        dict_match, exact_match = meta[idx]
        tag = assign_tag(score, dict_match, exact_match)

        preds.append(tm2_code)
        scores.append(score)
        tags.append(tag)

    results.append({
        "namc_code": row["namc_code"],
        "top1_pred": preds[0],
        "top1_score": scores[0],
        "top1_tag": tags[0],
        "top2_pred": preds[1],
        "top2_score": scores[1],
        "top2_tag": tags[1],
        "top3_pred": preds[2],
        "top3_score": scores[2],
        "top3_tag": tags[2],
    })

print(f"⏱️ Scoring time: {time.time() - t0:.2f}s")


# ==============================
# SAVE OUTPUT
# ==============================
mapping_df = pd.DataFrame(results)
mapping_df.to_csv("mapping_output.csv", index=False)

print("✅ Mapping saved")


# ==============================
# EVALUATION
# ==============================
merged = pd.merge(eval_df, mapping_df, on="namc_code", how="inner")

top1_correct = 0
top3_correct = 0

for _, r in merged.iterrows():
    true_code = r["tm2_code"]
    preds = [r["top1_pred"], r["top2_pred"], r["top3_pred"]]

    if true_code == preds[0]:
        top1_correct += 1

    if true_code in preds:
        top3_correct += 1

total = len(merged)

print("\n📊 RESULTS")
print("Top-1:", round(top1_correct / total, 4))
print("Top-3:", round(top3_correct / total, 4))
print("Coverage:", round(total / len(eval_df), 4))


# ==============================
# TAG DISTRIBUTION
# ==============================
all_tags = []
for col in ["top1_tag", "top2_tag", "top3_tag"]:
    all_tags.extend(mapping_df[col].tolist())

print("\n📊 TAG DISTRIBUTION")
print(pd.Series(all_tags).value_counts())


# ==============================
# TOTAL TIME
# ==============================
print(f"\n⏱️ TOTAL TIME: {time.time() - start_time:.2f}s")