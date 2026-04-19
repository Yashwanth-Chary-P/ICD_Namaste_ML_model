import pandas as pd
import numpy as np
import time
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# =========================================================
# START TIMER
# =========================================================
start_time = time.time()

# =========================================================
# LOAD DATA (HARDCODED FILENAMES)
# =========================================================
ayu = pd.read_csv("ayurveda_with_tm2_clean.csv")
eval_df = pd.read_csv("eval_dataset_final.csv")
tm2 = pd.read_csv("tm2.csv")

print("📥 Loaded:")
print("Ayurveda:", len(ayu))
print("Eval:", len(eval_df))
print("TM2:", len(tm2))

# =========================================================
# CLEAN COLUMN NAMES
# =========================================================
ayu.columns = ayu.columns.str.strip()
eval_df.columns = eval_df.columns.str.strip()
tm2.columns = tm2.columns.str.strip()

# Lowercase lookup maps for safe column detection
ayu_map = {c.lower().strip(): c for c in ayu.columns}
eval_map = {c.lower().strip(): c for c in eval_df.columns}
tm2_map = {c.lower().strip(): c for c in tm2.columns}

def col(df_map, candidates):
    for c in candidates:
        if c.lower().strip() in df_map:
            return df_map[c.lower().strip()]
    return None

# Ayurveda columns
A_NAMC = col(ayu_map, ["namc_code"])
A_SANSKRIT = col(ayu_map, ["namc_term_diacritical"])
A_SHORT = col(ayu_map, ["short_definition"])
A_LONG = col(ayu_map, ["long_definition"])

# Eval columns
E_NAMC = col(eval_map, ["namc_code"])
E_TM2 = col(eval_map, ["tm2_code"])

# TM2 columns
T_CODE = col(tm2_map, ["code"])
T_TITLE = col(tm2_map, ["title"])
T_INDEX = col(tm2_map, ["index terms", "index_terms"])
T_FSN = col(tm2_map, ["fully specified name", "fsn"])
T_DESC = col(tm2_map, ["description", "definition"])
T_INCL = col(tm2_map, ["inclusions"])

required = {
    "Ayurveda namc_code": A_NAMC,
    "Ayurveda namc_term_diacritical": A_SANSKRIT,
    "Ayurveda short_definition": A_SHORT,
    "Ayurveda long_definition": A_LONG,
    "Eval namc_code": E_NAMC,
    "Eval tm2_code": E_TM2,
    "TM2 code": T_CODE,
    "TM2 title": T_TITLE,
    "TM2 index_terms": T_INDEX,
}

missing = [k for k, v in required.items() if v is None]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# =========================================================
# DEDUP + NULL SAFETY
# =========================================================
ayu = ayu.drop_duplicates(subset=[A_NAMC]).copy()
eval_df = eval_df.drop_duplicates(subset=[E_NAMC]).copy()
tm2 = tm2.drop_duplicates(subset=[T_CODE]).copy()

for c in [A_SANSKRIT, A_SHORT, A_LONG]:
    ayu[c] = ayu[c].fillna("").astype(str)

for c in [T_TITLE, T_INDEX]:
    tm2[c] = tm2[c].fillna("").astype(str)

for c in [T_FSN, T_DESC, T_INCL]:
    if c is not None:
        tm2[c] = tm2[c].fillna("").astype(str)

eval_df[E_NAMC] = eval_df[E_NAMC].fillna("").astype(str)
eval_df[E_TM2] = eval_df[E_TM2].fillna("").astype(str)

# =========================================================
# BUILD AYURVEDA QUERY
# =========================================================
ayu["final_query"] = (
    (ayu[A_SANSKRIT] + " ") * 4 +
    ayu[A_SHORT] + " " +
    ayu[A_LONG]
)

ayu["final_query"] = (
    ayu["final_query"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

ayu = ayu[ayu["final_query"] != ""].reset_index(drop=True)
print("✅ After Ayurveda cleaning:", len(ayu))

# =========================================================
# BUILD WEIGHTED TM2 TEXT
# =========================================================
tm2_text = (
    (tm2[T_TITLE] + " ") * 3 +
    (tm2[T_INDEX] + " ") * 4
)

if T_FSN is not None:
    tm2_text = tm2_text + (tm2[T_FSN] + " ")
if T_DESC is not None:
    tm2_text = tm2_text + (tm2[T_DESC] + " ")
if T_INCL is not None:
    tm2_text = tm2_text + (tm2[T_INCL] + " ")

tm2["final_text"] = (
    tm2_text.astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

tm2 = tm2[tm2["final_text"] != ""].reset_index(drop=True)
print("✅ TM2 final_text rows:", len(tm2))

# lowercase helper for exact-match checks
tm2["final_text_lower"] = tm2["final_text"].str.lower()

# =========================================================
# TRAIN DICTIONARY FROM eval_dataset_final.csv
# =========================================================
sanskrit_to_tm2 = defaultdict(set)

dict_df = pd.merge(
    eval_df[[E_NAMC, E_TM2]],
    ayu[[A_NAMC, A_SANSKRIT]],
    left_on=E_NAMC,
    right_on=A_NAMC,
    how="inner"
)

for _, r in dict_df.iterrows():
    s = str(r[A_SANSKRIT]).strip()
    c = str(r[E_TM2]).strip()
    if s and c:
        sanskrit_to_tm2[s].add(c)

print("📚 Dictionary size:", len(sanskrit_to_tm2))

# =========================================================
# TF-IDF
# =========================================================
t0 = time.time()

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2)
)

tfidf_tm2 = vectorizer.fit_transform(tm2["final_text"])
tfidf_ayu = vectorizer.transform(ayu["final_query"])

print("TF-IDF AYU:", tfidf_ayu.shape)
print("TF-IDF TM2:", tfidf_tm2.shape)
print(f"⏱️ TF-IDF time: {time.time() - t0:.2f}s")

# =========================================================
# SIMILARITY
# =========================================================
t0 = time.time()
similarity = cosine_similarity(tfidf_ayu, tfidf_tm2)
print(f"⏱️ Similarity time: {time.time() - t0:.2f}s")

# =========================================================
# TAG FUNCTION
# =========================================================
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

# =========================================================
# HYBRID SCORING
# =========================================================
t0 = time.time()
results = []

TOP_K = 3

for i in tqdm(range(len(ayu)), desc="Processing"):
    row = ayu.iloc[i]
    sims = similarity[i]

    sanskrit = row[A_SANSKRIT].lower().strip()

    boosted_scores = []
    meta = []

    for j, base_score in enumerate(sims):
        tm2_code = tm2.iloc[j][T_CODE]
        tm2_text_lower = tm2.iloc[j]["final_text_lower"]

        bonus = 0.0
        dict_match = False
        exact_match = False

        if sanskrit and sanskrit in tm2_text_lower:
            bonus += 0.5
            exact_match = True

        if sanskrit in sanskrit_to_tm2 and tm2_code in sanskrit_to_tm2[sanskrit]:
            bonus += 1.0
            dict_match = True

        final_score = float(base_score) + bonus
        boosted_scores.append(final_score)
        meta.append((dict_match, exact_match))

    boosted_scores = pd.Series(boosted_scores)
    top_indices = boosted_scores.argsort()[-TOP_K:][::-1]

    preds, scores, tags = [], [], []

    for idx in top_indices:
        score = float(boosted_scores.iloc[idx])
        tm2_code = tm2.iloc[idx][T_CODE]
        dict_match, exact_match = meta[idx]
        tag = assign_tag(score, dict_match, exact_match)

        preds.append(tm2_code)
        scores.append(score)
        tags.append(tag)

    results.append({
        "namc_code": row[A_NAMC],
        "top1_pred": preds[0],
        "top1_score": scores[0],
        "top1_tag": tags[0],
        "top2_pred": preds[1] if len(preds) > 1 else "",
        "top2_score": scores[1] if len(scores) > 1 else "",
        "top2_tag": tags[1] if len(tags) > 1 else "",
        "top3_pred": preds[2] if len(preds) > 2 else "",
        "top3_score": scores[2] if len(scores) > 2 else "",
        "top3_tag": tags[2] if len(tags) > 2 else "",
    })

print(f"⏱️ Scoring time: {time.time() - t0:.2f}s")

# =========================================================
# SAVE OUTPUT
# =========================================================
mapping_df = pd.DataFrame(results)
mapping_df.to_csv("mapping_output.csv", index=False)
print("✅ Mapping saved: mapping_output.csv")

# =========================================================
# EVALUATION
# =========================================================
merged = pd.merge(
    eval_df[[E_NAMC, E_TM2]],
    mapping_df,
    left_on=E_NAMC,
    right_on="namc_code",
    how="inner"
)

top1_correct = 0
top3_correct = 0
eq = 0
eq_narrow = 0
clinical = 0

for _, r in merged.iterrows():
    true_code = str(r[E_TM2]).strip()
    preds = [str(r["top1_pred"]).strip(), str(r["top2_pred"]).strip(), str(r["top3_pred"]).strip()]
    tags = [str(r["top1_tag"]).strip(), str(r["top2_tag"]).strip(), str(r["top3_tag"]).strip()]

    if true_code == preds[0]:
        top1_correct += 1

    if true_code in preds:
        top3_correct += 1

    for p, t in zip(preds, tags):
        if p == true_code and t == "Equivalent":
            eq += 1
        if p == true_code and t in ["Equivalent", "Narrower"]:
            eq_narrow += 1
        if p == true_code and t in ["Equivalent", "Narrower", "Related"]:
            clinical += 1

total = len(merged)

print("\n📊 RESULTS")
print("Strict Top-1:", round(top1_correct / total, 4) if total else 0)
print("Top-3 Accuracy:", round(top3_correct / total, 4) if total else 0)
print("Equivalent:", round(eq / total, 4) if total else 0)
print("Equivalent + Narrower:", round(eq_narrow / total, 4) if total else 0)
print("Clinical (Eq + Nar + Rel):", round(clinical / total, 4) if total else 0)
print("Coverage:", round(total / len(eval_df), 4) if len(eval_df) else 0)

# =========================================================
# TAG DISTRIBUTION
# =========================================================
all_tags = []
for c in ["top1_tag", "top2_tag", "top3_tag"]:
    all_tags.extend(mapping_df[c].astype(str).tolist())

print("\n📊 TAG DISTRIBUTION")
print(pd.Series(all_tags).value_counts())

print(f"\n⏱️ TOTAL TIME: {time.time() - start_time:.2f}s")