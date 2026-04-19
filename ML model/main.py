import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# =========================================================
# FILES
# =========================================================
AYU_FILE = "ayurveda_with_tm2_clean.csv"
EVAL_FILE = "eval_dataset_final.csv"
TM2_FILE = "tm2.csv"
OUTPUT_FILE = "mapping_output_no_dict_tagged.csv"

# =========================================================
# TIMER
# =========================================================
start_time = time.time()

# =========================================================
# LOAD
# =========================================================
ayu = pd.read_csv(AYU_FILE)
eval_df = pd.read_csv(EVAL_FILE)
tm2 = pd.read_csv(TM2_FILE)

print("📥 Loaded:")
print("Ayurveda:", len(ayu))
print("Eval:", len(eval_df))
print("TM2:", len(tm2))

# =========================================================
# STANDARDIZE COLUMN NAMES
# =========================================================
ayu.columns = ayu.columns.str.strip().str.lower()
eval_df.columns = eval_df.columns.str.strip().str.lower()
tm2.columns = tm2.columns.str.strip().str.lower()

# Required columns
A_NAMC = "namc_code"
A_SANSKRIT = "namc_term_diacritical"
A_SHORT = "short_definition"
A_LONG = "long_definition"

E_NAMC = "namc_code"
E_TM2 = "tm2_code"

T_CODE = "code"
T_TITLE = "title"
T_INDEX = "index terms"

# Optional TM2 columns if present
optional_tm2_cols = {
    "fully specified name": "fsn",
    "description": "definition",
    "inclusions": "inclusions",
    "exclusions": "exclusions",
}

for src, dst in optional_tm2_cols.items():
    if src in tm2.columns and dst not in tm2.columns:
        tm2[dst] = tm2[src]

# Ensure required columns exist
for col in [A_NAMC, A_SANSKRIT, A_SHORT, A_LONG]:
    if col not in ayu.columns:
        raise ValueError(f"Missing Ayurveda column: {col}")

for col in [E_NAMC, E_TM2]:
    if col not in eval_df.columns:
        raise ValueError(f"Missing eval column: {col}")

for col in [T_CODE, T_TITLE, T_INDEX]:
    if col not in tm2.columns:
        raise ValueError(f"Missing TM2 column: {col}")

# =========================================================
# CLEAN + DEDUP
# =========================================================
ayu = ayu.drop_duplicates(subset=[A_NAMC]).copy()
eval_df = eval_df.drop_duplicates(subset=[E_NAMC]).copy()
tm2 = tm2.drop_duplicates(subset=[T_CODE]).copy()

for c in [A_SANSKRIT, A_SHORT, A_LONG]:
    ayu[c] = ayu[c].fillna("").astype(str)

for c in [T_TITLE, T_INDEX]:
    tm2[c] = tm2[c].fillna("").astype(str)

for c in ["fsn", "definition", "inclusions", "exclusions"]:
    if c not in tm2.columns:
        tm2[c] = ""
    tm2[c] = tm2[c].fillna("").astype(str)

eval_df[E_NAMC] = eval_df[E_NAMC].fillna("").astype(str)
eval_df[E_TM2] = eval_df[E_TM2].fillna("").astype(str)

# Keep only rows with a usable query
ayu = ayu[ayu[A_NAMC].astype(str).str.strip() != ""].reset_index(drop=True)

# =========================================================
# BUILD AYURVEDA QUERY
# =========================================================
# Sanskrit is preserved, no aggressive cleaning, no diacritic removal.
# Weighted query follows your methodology: Sanskrit + definitions.
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
# BUILD TM2 TEXT
# =========================================================
# No dictionary. No eval labels. Just weighted TM2 text.
# title + index terms carry the strongest signal.
tm2["final_text"] = (
    (tm2[T_TITLE] + " ") * 3 +
    (tm2[T_INDEX] + " ") * 4 +
    (tm2["fsn"] + " ") * 1 +
    (tm2["definition"] + " ") * 1 +
    (tm2["inclusions"] + " ") * 1
)

tm2["final_text"] = (
    tm2["final_text"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

tm2 = tm2[tm2["final_text"] != ""].reset_index(drop=True)
tm2["final_text_lower"] = tm2["final_text"].str.lower()

print("✅ TM2 final_text rows:", len(tm2))

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
def assign_tag(score, exact_match=False):
    if exact_match and score >= 0.55:
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
# MAPPING
# =========================================================
t0 = time.time()
results = []

TOP_K = 5  # requested top-5

for i in tqdm(range(len(ayu)), desc="Processing"):
    row = ayu.iloc[i]
    sims = similarity[i]
    sanskrit = row[A_SANSKRIT].lower().strip()

    # Candidate pruning for speed and to reduce noise
    top_candidates = np.argsort(sims)[-50:]

    reranked = []

    for j in top_candidates:
        base_score = float(sims[j])
        tm2_text_lower = tm2.iloc[j]["final_text_lower"]

        score = base_score

        # No dictionary. Only exact Sanskrit term presence boost.
        exact_match = False
        if sanskrit and sanskrit in tm2_text_lower:
            score += 0.5
            exact_match = True

        # Small bonus for overlap with the main searchable field
        # (kept intentionally light so the model still reflects the similarity search)
        score += 0.25 * base_score

        reranked.append((j, score, exact_match))

    reranked = sorted(reranked, key=lambda x: x[1], reverse=True)[:TOP_K]

    row_out = {
        "namc_code": row[A_NAMC]
    }

    for k, (j, score, exact_match) in enumerate(reranked, start=1):
        pred_code = tm2.iloc[j][T_CODE]
        tag = assign_tag(score, exact_match=exact_match)

        row_out[f"top{k}_pred"] = pred_code
        row_out[f"top{k}_score"] = score
        row_out[f"top{k}_tag"] = tag

    # Fill missing top positions if any
    for k in range(len(reranked) + 1, TOP_K + 1):
        row_out[f"top{k}_pred"] = ""
        row_out[f"top{k}_score"] = 0.0
        row_out[f"top{k}_tag"] = "Weak"

    results.append(row_out)

print(f"⏱️ Scoring time: {time.time() - t0:.2f}s")

# =========================================================
# SAVE OUTPUT
# =========================================================
mapping_df = pd.DataFrame(results)
mapping_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Mapping saved: {OUTPUT_FILE}")

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

strict_top1 = 0
top3 = 0
top5 = 0
equivalent = 0
equivalent_narrower = 0
clinical = 0

for _, r in merged.iterrows():
    true_code = str(r[E_TM2]).strip()

    preds = [str(r[f"top{i}_pred"]).strip() for i in range(1, TOP_K + 1)]
    tags = [str(r[f"top{i}_tag"]).strip() for i in range(1, TOP_K + 1)]

    if true_code == preds[0]:
        strict_top1 += 1

    if true_code in preds[:3]:
        top3 += 1

    if true_code in preds:
        top5 += 1

    for p, t in zip(preds, tags):
        if p == true_code and t == "Equivalent":
            equivalent += 1
        if p == true_code and t in ["Equivalent", "Narrower"]:
            equivalent_narrower += 1
        if p == true_code and t in ["Equivalent", "Narrower", "Related"]:
            clinical += 1

total = len(merged)

print("\n📊 RESULTS (NO DICTIONARY)")
print("Strict Top-1:", round(strict_top1 / total, 4) if total else 0)
print("Top-3 Accuracy:", round(top3 / total, 4) if total else 0)
print("Top-5 Accuracy:", round(top5 / total, 4) if total else 0)
print("Equivalent:", round(equivalent / total, 4) if total else 0)
print("Equivalent + Narrower:", round(equivalent_narrower / total, 4) if total else 0)
print("Clinical (Eq + Nar + Rel):", round(clinical / total, 4) if total else 0)
print("Coverage:", round(total / len(eval_df), 4) if len(eval_df) else 0)

# =========================================================
# TAG DISTRIBUTION
# =========================================================
all_tags = []
for col in [f"top{i}_tag" for i in range(1, TOP_K + 1)]:
    if col in mapping_df.columns:
        all_tags.extend(mapping_df[col].astype(str).tolist())

print("\n📊 TAG DISTRIBUTION")
print(pd.Series(all_tags).value_counts())

print(f"\n⏱️ TOTAL TIME: {time.time() - start_time:.2f}s")