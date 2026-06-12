import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Optional BERT
try:
    import torch
    from sentence_transformers import SentenceTransformer, util
    HAVE_BERT = True
except:
    HAVE_BERT = False

# =========================================================
# FILES
# =========================================================
AYU_FILE = "ayurveda_with_tm2_clean.csv"
EVAL_FILE = "eval_dataset_final.csv"
TM2_FILE = "tm2.csv"

TOP_K = 5

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
# CLEAN
# =========================================================
ayu.columns = ayu.columns.str.lower().str.strip()
eval_df.columns = eval_df.columns.str.lower().str.strip()
tm2.columns = tm2.columns.str.lower().str.strip()

A_NAMC = "namc_code"
A_SANS = "namc_term_diacritical"
A_SHORT = "short_definition"
A_LONG = "long_definition"

E_NAMC = "namc_code"
E_TM2 = "tm2_code"

T_CODE = "code"
T_TITLE = "title"
T_INDEX = "index terms"

for col in [A_SANS, A_SHORT, A_LONG]:
    ayu[col] = ayu[col].fillna("").astype(str)

tm2[T_TITLE] = tm2[T_TITLE].fillna("").astype(str)
tm2[T_INDEX] = tm2[T_INDEX].fillna("").astype(str)

# =========================================================
# QUERY
# =========================================================
ayu["final_query"] = (
    (ayu[A_SANS] + " ") * 4 +
    ayu[A_SHORT] + " " +
    ayu[A_LONG]
).str.replace(r"\s+", " ", regex=True).str.strip()

# =========================================================
# TM2 TEXT
# =========================================================
tm2["final_text"] = (
    (tm2[T_TITLE] + " ") * 3 +
    (tm2[T_INDEX] + " ") * 4
).str.replace(r"\s+", " ", regex=True).str.strip()

tm2["final_text_lower"] = tm2["final_text"].str.lower()

# =========================================================
# SPLIT
# =========================================================
train_eval, test_eval = train_test_split(eval_df, test_size=0.3, random_state=42)

true_map = dict(zip(
    test_eval[E_NAMC].astype(str).str.strip(),
    test_eval[E_TM2].astype(str).str.strip()
))

ayu["namc_code"] = ayu["namc_code"].astype(str).str.strip()

ayu_test = ayu[ayu["namc_code"].isin(true_map.keys())].copy()
ayu_test["true_tm2_code"] = ayu_test["namc_code"].map(true_map)

print("\n📊 Split summary:")
print("Test rows:", len(ayu_test))

# =========================================================
# TF-IDF
# =========================================================
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
vectorizer.fit(pd.concat([tm2["final_text"], ayu["final_query"]]))

tfidf_tm2 = vectorizer.transform(tm2["final_text"])
tfidf_test = vectorizer.transform(ayu_test["final_query"])

similarity = cosine_similarity(tfidf_test, tfidf_tm2)

# =========================================================
# TAGGING (FIXED)
# =========================================================
def assign_tag(score, exact):
    if exact:
        return "Equivalent"
    if score >= 0.70:
        return "Equivalent"
    elif score >= 0.50:
        return "Narrower"
    elif score >= 0.30:
        return "Related"
    return "Weak"

# =========================================================
# PREDICT (WITH NORMALIZATION)
# =========================================================
def predict(scores, name, boost=False):
    results = []
    tm2_lower = tm2["final_text_lower"].tolist()

    for i in tqdm(range(len(ayu_test)), desc=name):
        row = ayu_test.iloc[i]
        sims = scores[i].copy()

        # 🔥 NORMALIZE SCORES (IMPORTANT)
        sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)

        sans = row[A_SANS].lower()

        if boost and sans:
            for j in range(len(sims)):
                if sans in tm2_lower[j]:
                    sims[j] += 0.25

        top_idx = np.argsort(sims)[-TOP_K:][::-1]

        rec = {
            "namc_code": row["namc_code"],
            "true_tm2_code": row["true_tm2_code"],
            "algorithm": name
        }

        for k, j in enumerate(top_idx):
            pred = tm2.iloc[j][T_CODE]
            score = sims[j]
            exact = sans in tm2_lower[j]
            tag = assign_tag(score, exact)

            rec[f"top{k+1}_pred"] = pred
            rec[f"top{k+1}_score"] = score
            rec[f"top{k+1}_tag"] = tag

        results.append(rec)

    return pd.DataFrame(results)

# =========================================================
# RUN MODELS
# =========================================================
pred_tfidf = predict(similarity, "TFIDF")
pred_boost = predict(similarity, "TFIDF+BOOST", boost=True)

# =========================================================
# BERT
# =========================================================
if HAVE_BERT:
    print("\n🧠 Running BERT...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    emb_tm2 = model.encode(tm2["final_text"].tolist(), convert_to_tensor=True)
    emb_ayu = model.encode(ayu_test["final_query"].tolist(), convert_to_tensor=True)

    bert_sim = util.cos_sim(emb_ayu, emb_tm2).cpu().numpy()

    pred_bert = predict(bert_sim, "BERT")

    # HYBRID
    norm_tfidf = (similarity - similarity.min()) / (similarity.max() - similarity.min() + 1e-9)
    norm_bert = (bert_sim - bert_sim.min()) / (bert_sim.max() - bert_sim.min() + 1e-9)

    hybrid = 0.5 * norm_tfidf + 0.5 * norm_bert
    pred_hybrid = predict(hybrid, "HYBRID", boost=True)
else:
    pred_bert = None
    pred_hybrid = None

# =========================================================
# EVALUATION
# =========================================================
def evaluate(df):
    return {
        "Top1": (df["true_tm2_code"] == df["top1_pred"]).mean(),
        "Top3": df.apply(lambda r: r["true_tm2_code"] in [r["top1_pred"], r["top2_pred"], r["top3_pred"]], axis=1).mean(),
        "Top5": df.apply(lambda r: r["true_tm2_code"] in [r[f"top{i}_pred"] for i in range(1,6)], axis=1).mean()
    }

print("\n📊 FINAL RESULTS")

models = [pred_tfidf, pred_boost, pred_bert, pred_hybrid]

for df in models:
    if df is None:
        continue
    print(f"\n🔹 {df['algorithm'].iloc[0]}")
    print(evaluate(df))

# =========================================================
# TAG DISTRIBUTION
# =========================================================
def tag_distribution(df, name):
    if df is None:
        return

    tags = []
    for col in [f"top{i}_tag" for i in range(1,6)]:
        tags.extend(df[col].tolist())

    counts = pd.Series(tags).value_counts()
    percent = (counts / counts.sum() * 100).round(2)

    print(f"\n📊 TAG DISTRIBUTION — {name}")
    print(pd.DataFrame({
        "Count": counts,
        "Percentage (%)": percent
    }))

tag_distribution(pred_tfidf, "TFIDF")
tag_distribution(pred_boost, "TFIDF+BOOST")

if HAVE_BERT:
    tag_distribution(pred_bert, "BERT")
    tag_distribution(pred_hybrid, "HYBRID")

print("\n⏱️ Done")