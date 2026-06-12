import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# Optional BERT
try:
    from sentence_transformers import SentenceTransformer, util
    HAVE_BERT = True
except Exception:
    HAVE_BERT = False

# =========================================================
# CONFIG
# =========================================================
AYU_FILE = "ayurveda_with_tm2_clean.csv"
EVAL_FILE = "eval_dataset_final.csv"
TM2_FILE = "tm2.csv"

TOP_K = 5
TEST_SIZE = 0.30
RANDOM_STATE = 42
RERANK_BOOST = 0.25

# =========================================================
# LOAD
# =========================================================
ayu = pd.read_csv(AYU_FILE)
eval_df = pd.read_csv(EVAL_FILE)
tm2 = pd.read_csv(TM2_FILE)

print("📥 Loaded:")
print(f"  Ayurveda: {len(ayu)}")
print(f"  Eval:      {len(eval_df)}")
print(f"  TM2:       {len(tm2)}")

# =========================================================
# CLEAN COLUMN NAMES
# =========================================================
ayu.columns = ayu.columns.str.lower().str.strip()
eval_df.columns = eval_df.columns.str.lower().str.strip()
tm2.columns = tm2.columns.str.lower().str.strip()

# =========================================================
# COLUMN NAMES
# =========================================================
A_NAMC = "namc_code"
A_SANS = "namc_term_diacritical"
A_SHORT = "short_definition"
A_LONG = "long_definition"

E_NAMC = "namc_code"
E_TM2 = "tm2_code"

T_CODE = "code"
T_TITLE = "title"
T_INDEX = "index terms"

# =========================================================
# BASIC VALIDATION
# =========================================================
required_ayu_cols = [A_NAMC, A_SANS, A_SHORT, A_LONG]
required_eval_cols = [E_NAMC, E_TM2]
required_tm2_cols = [T_CODE, T_TITLE, T_INDEX]

for col in required_ayu_cols:
    if col not in ayu.columns:
        raise ValueError(f"Missing column in Ayurveda file: {col}")

for col in required_eval_cols:
    if col not in eval_df.columns:
        raise ValueError(f"Missing column in eval file: {col}")

for col in required_tm2_cols:
    if col not in tm2.columns:
        raise ValueError(f"Missing column in TM2 file: {col}")

# =========================================================
# FILL NA AND STRING CONVERSION
# =========================================================
for col in [A_NAMC, A_SANS, A_SHORT, A_LONG]:
    ayu[col] = ayu[col].fillna("").astype(str)

for col in [E_NAMC, E_TM2]:
    eval_df[col] = eval_df[col].fillna("").astype(str)

for col in [T_CODE, T_TITLE, T_INDEX]:
    tm2[col] = tm2[col].fillna("").astype(str)

# strip spaces
ayu[A_NAMC] = ayu[A_NAMC].str.strip()
ayu[A_SANS] = ayu[A_SANS].str.strip()
ayu[A_SHORT] = ayu[A_SHORT].str.strip()
ayu[A_LONG] = ayu[A_LONG].str.strip()

eval_df[E_NAMC] = eval_df[E_NAMC].str.strip()
eval_df[E_TM2] = eval_df[E_TM2].str.strip()

tm2[T_CODE] = tm2[T_CODE].str.strip()
tm2[T_TITLE] = tm2[T_TITLE].str.strip()
tm2[T_INDEX] = tm2[T_INDEX].str.strip()

# =========================================================
# TEXT BUILDING
# =========================================================
def normalize_spaces(text: pd.Series) -> pd.Series:
    return text.str.replace(r"\s+", " ", regex=True).str.strip()

# Ayurveda query: preserve Sanskrit and weight it heavily
ayu["final_query"] = normalize_spaces(
    (ayu[A_SANS] + " ") * 4 +
    ayu[A_SHORT] + " " +
    ayu[A_LONG]
)

# TM2 text: title + index terms
tm2["final_text"] = normalize_spaces(
    (tm2[T_TITLE] + " ") * 3 +
    (tm2[T_INDEX] + " ") * 4
)

tm2["final_text_lower"] = tm2["final_text"].str.lower()

# =========================================================
# SPLIT FOR VALID EVALUATION
# =========================================================
train_eval, test_eval = train_test_split(
    eval_df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_codes = set(train_eval[E_NAMC].astype(str).str.strip())
test_codes = set(test_eval[E_NAMC].astype(str).str.strip())

# Ayurveda rows corresponding to train/test codes
ayu_train = ayu[ayu[A_NAMC].isin(train_codes)].copy()
ayu_test = ayu[ayu[A_NAMC].isin(test_codes)].copy()

# Attach ground truth to test rows
true_map = dict(
    zip(
        test_eval[E_NAMC].astype(str).str.strip(),
        test_eval[E_TM2].astype(str).str.strip()
    )
)
ayu_test["true_tm2_code"] = ayu_test[A_NAMC].map(true_map)

# Remove any rows where mapping is missing
ayu_test = ayu_test.dropna(subset=["true_tm2_code"]).copy()
ayu_test["true_tm2_code"] = ayu_test["true_tm2_code"].astype(str).str.strip()

print("\n📊 Split summary:")
print(f"  Train eval codes: {len(train_codes)}")
print(f"  Test eval codes:  {len(test_codes)}")
print(f"  Ayurveda train rows: {len(ayu_train)}")
print(f"  Ayurveda test rows:  {len(ayu_test)}")

# =========================================================
# HELPERS
# =========================================================
def tokenize(text: str):
    return str(text).lower().split()

def minmax_norm(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mn = arr.min()
    mx = arr.max()
    return (arr - mn) / (mx - mn + 1e-9)

def assign_tag(score: float, exact: bool) -> str:
    if exact:
        return "Equivalent"
    if score >= 0.70:
        return "Equivalent"
    elif score >= 0.50:
        return "Narrower"
    elif score >= 0.30:
        return "Related"
    return "Weak"

def evaluate_predictions(df: pd.DataFrame):
    top1 = (df["true_tm2_code"] == df["top1_pred"]).mean()
    top3 = df.apply(
        lambda r: r["true_tm2_code"] in [r["top1_pred"], r["top2_pred"], r["top3_pred"]],
        axis=1
    ).mean()
    top5 = df.apply(
        lambda r: r["true_tm2_code"] in [r[f"top{i}_pred"] for i in range(1, 6)],
        axis=1
    ).mean()

    return {
        "Top1": float(top1),
        "Top3": float(top3),
        "Top5": float(top5),
    }

def tag_distribution(df: pd.DataFrame, name: str):
    tags = []
    for col in [f"top{i}_tag" for i in range(1, 6)]:
        tags.extend(df[col].tolist())

    counts = pd.Series(tags).value_counts()
    percent = (counts / counts.sum() * 100).round(2)

    print(f"\n📊 TAG DISTRIBUTION — {name}")
    print(pd.DataFrame({"Count": counts, "Percentage (%)": percent}))

def predict_from_scores(scores: np.ndarray, name: str, rerank: bool = False) -> pd.DataFrame:
    results = []
    tm2_lower = tm2["final_text_lower"].tolist()

    for i in tqdm(range(len(ayu_test)), desc=name):
        row = ayu_test.iloc[i]
        sims = np.array(scores[i], dtype=float)

        # normalize scores for stable ranking and thresholding
        sims = minmax_norm(sims)

        sans = row[A_SANS].lower().strip()

        # lexical re-ranking using Sanskrit exact substring match
        if rerank and sans:
            for j in range(len(sims)):
                if sans and sans in tm2_lower[j]:
                    sims[j] += RERANK_BOOST

        top_idx = np.argsort(sims)[-TOP_K:][::-1]

        rec = {
            "namc_code": row[A_NAMC],
            "true_tm2_code": row["true_tm2_code"],
            "algorithm": name,
        }

        for k, j in enumerate(top_idx, start=1):
            pred = tm2.iloc[j][T_CODE]
            score = float(sims[j])
            exact = bool(sans and sans in tm2_lower[j])
            tag = assign_tag(score, exact)

            rec[f"top{k}_pred"] = pred
            rec[f"top{k}_score"] = score
            rec[f"top{k}_tag"] = tag

        results.append(rec)

    return pd.DataFrame(results)

# =========================================================
# TF-IDF BASELINE
# NOTE: fit on TM2 + TRAIN Ayurveda only (avoids leakage)
# =========================================================
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))

vectorizer.fit(pd.concat([tm2["final_text"], ayu_train["final_query"]], ignore_index=True))

tfidf_tm2 = vectorizer.transform(tm2["final_text"])
tfidf_test = vectorizer.transform(ayu_test["final_query"])

tfidf_sim = cosine_similarity(tfidf_test, tfidf_tm2)

pred_tfidf = predict_from_scores(tfidf_sim, "TFIDF", rerank=False)
pred_tfidf_rerank = predict_from_scores(tfidf_sim, "TFIDF+RERANK", rerank=True)

# =========================================================
# BM25 BASELINE
# =========================================================
tm2_tokens = [tokenize(text) for text in tm2["final_text"].tolist()]
bm25 = BM25Okapi(tm2_tokens)

bm25_scores_list = []
for q in tqdm(ayu_test["final_query"].tolist(), desc="BM25"):
    q_tokens = tokenize(q)
    bm25_scores_list.append(bm25.get_scores(q_tokens))

bm25_scores = np.array(bm25_scores_list)

pred_bm25 = predict_from_scores(bm25_scores, "BM25", rerank=False)
pred_bm25_rerank = predict_from_scores(bm25_scores, "BM25+RERANK", rerank=True)

# =========================================================
# OPTIONAL BERT
# =========================================================
pred_bert = None
pred_hybrid = None

if HAVE_BERT:
    print("\n🧠 Running BERT...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    emb_tm2 = model.encode(tm2["final_text"].tolist(), convert_to_tensor=True)
    emb_ayu = model.encode(ayu_test["final_query"].tolist(), convert_to_tensor=True)

    bert_sim = util.cos_sim(emb_ayu, emb_tm2).cpu().numpy()

    pred_bert = predict_from_scores(bert_sim, "BERT", rerank=False)

    # Hybrid = TF-IDF + BERT
    tfidf_norm = minmax_norm(tfidf_sim)
    bert_norm = minmax_norm(bert_sim)
    hybrid_scores = 0.5 * tfidf_norm + 0.5 * bert_norm

    pred_hybrid = predict_from_scores(hybrid_scores, "HYBRID", rerank=True)

# =========================================================
# RESULTS
# =========================================================
models = [
    pred_tfidf,
    pred_tfidf_rerank,
    pred_bm25,
    pred_bm25_rerank,
    pred_bert,
    pred_hybrid,
]

print("\n📊 FINAL RESULTS")
summary_rows = []

for df in models:
    if df is None:
        continue
    metrics = evaluate_predictions(df)
    algo = df["algorithm"].iloc[0]
    print(f"\n🔹 {algo}")
    print(metrics)
    summary_rows.append({"algorithm": algo, **metrics})

summary_df = pd.DataFrame(summary_rows)
print("\n=== SUMMARY TABLE ===")
print(summary_df)

# =========================================================
# TAG DISTRIBUTION
# =========================================================
tag_distribution(pred_tfidf, "TFIDF")
tag_distribution(pred_tfidf_rerank, "TFIDF+RERANK")
tag_distribution(pred_bm25, "BM25")
tag_distribution(pred_bm25_rerank, "BM25+RERANK")

if pred_bert is not None:
    tag_distribution(pred_bert, "BERT")
if pred_hybrid is not None:
    tag_distribution(pred_hybrid, "HYBRID")

# =========================================================
# SAVE OUTPUTS
# =========================================================
pred_tfidf.to_csv("predictions_tfidf.csv", index=False)
pred_tfidf_rerank.to_csv("predictions_tfidf_rerank.csv", index=False)
pred_bm25.to_csv("predictions_bm25.csv", index=False)
pred_bm25_rerank.to_csv("predictions_bm25_rerank.csv", index=False)

if pred_bert is not None:
    pred_bert.to_csv("predictions_bert.csv", index=False)
if pred_hybrid is not None:
    pred_hybrid.to_csv("predictions_hybrid.csv", index=False)

summary_df.to_csv("comparison_summary.csv", index=False)

print("\n⏱️ Done")
print("Saved: predictions_tfidf.csv, predictions_tfidf_rerank.csv, predictions_bm25.csv, predictions_bm25_rerank.csv, comparison_summary.csv")