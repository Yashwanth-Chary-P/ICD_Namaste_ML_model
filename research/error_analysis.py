
import re
from pathlib import Path

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

OUTPUT_DIR = Path("outputs_error_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# Helpful lookup columns
tm2["tm2_row_id"] = np.arange(len(tm2))
tm2_lookup = tm2.set_index(T_CODE, drop=False)

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

def safe_get_tm2(code: str):
    code = str(code).strip()
    if code in tm2_lookup.index:
        row = tm2_lookup.loc[code]
        # If duplicate codes exist, keep the first row deterministically
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row
    return None

def compute_token_overlap(a: str, b: str) -> int:
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    return len(a_tokens & b_tokens)

def candidate_detail(row, cand_idx: int, score: float, rank: int, sans: str):
    cand = tm2.iloc[cand_idx]
    cand_text = cand["final_text"]
    cand_lower = cand["final_text_lower"]
    exact = bool(sans and sans in cand_lower)

    return {
        "rank": rank,
        "cand_row_id": int(cand["tm2_row_id"]),
        "pred_tm2_code": cand[T_CODE],
        "pred_tm2_title": cand[T_TITLE],
        "pred_tm2_index_terms": cand[T_INDEX],
        "pred_tm2_text": cand_text,
        "pred_score": float(score),
        "pred_tag": assign_tag(float(score), exact),
        "pred_exact_sanskrit_match": exact,
        "query_token_overlap": compute_token_overlap(row["final_query"], cand_text),
        "query_vs_pred_sans_overlap": compute_token_overlap(row[A_SANS], cand_text),
    }

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
        if col in df.columns:
            tags.extend(df[col].tolist())

    counts = pd.Series(tags).value_counts()
    percent = (counts / counts.sum() * 100).round(2)

    print(f"\n📊 TAG DISTRIBUTION — {name}")
    print(pd.DataFrame({"Count": counts, "Percentage (%)": percent}))

def model_error_summary(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    total = len(df)
    correct_top1 = (df["top1_correct"]).sum()
    top5_hit = (df["top5_hit"]).sum()
    rerank_gain = int(df.get("rerank_fixed", pd.Series([False] * total)).sum())
    rerank_loss = int(df.get("rerank_hurt", pd.Series([False] * total)).sum())

    rows.append({"metric": "total_queries", "value": total, "model": model_name})
    rows.append({"metric": "top1_correct", "value": int(correct_top1), "model": model_name})
    rows.append({"metric": "top1_error", "value": int(total - correct_top1), "model": model_name})
    rows.append({"metric": "top5_hit", "value": int(top5_hit), "model": model_name})
    rows.append({"metric": "top5_miss", "value": int(total - top5_hit), "model": model_name})
    rows.append({"metric": "rerank_fixed", "value": rerank_gain, "model": model_name})
    rows.append({"metric": "rerank_hurt", "value": rerank_loss, "model": model_name})

    err_df = pd.DataFrame(rows)
    return err_df

def classify_error(row: pd.Series) -> str:
    if row["top1_correct"]:
        return "correct_top1"
    if row["top5_hit"]:
        if row.get("true_rank", np.nan) == 2:
            return "top2_miss"
        if row.get("true_rank", np.nan) == 3:
            return "top3_miss"
        if row.get("true_rank", np.nan) == 4:
            return "top4_miss"
        if row.get("true_rank", np.nan) == 5:
            return "top5_miss"
        return "top5_hit_nontrivial"
    return "top5_miss"

def build_long_form(df_wide: pd.DataFrame, model_name: str) -> pd.DataFrame:
    records = []
    for _, row in df_wide.iterrows():
        base = {
            "algorithm": model_name,
            "namc_code": row["namc_code"],
            "true_tm2_code": row["true_tm2_code"],
            "query_text": row["query_text"],
            "query_sanskrit_term": row["query_sanskrit_term"],
            "top1_correct": row["top1_correct"],
            "top5_hit": row["top5_hit"],
            "true_rank": row["true_rank"],
        }
        for k in range(1, TOP_K + 1):
            rec = base.copy()
            rec["rank"] = k
            rec["pred_tm2_code"] = row[f"top{k}_pred"]
            rec["pred_tm2_title"] = row[f"top{k}_title"]
            rec["pred_tm2_index_terms"] = row[f"top{k}_index_terms"]
            rec["pred_score"] = row[f"top{k}_score"]
            rec["pred_tag"] = row[f"top{k}_tag"]
            rec["pred_correct"] = (row[f"top{k}_pred"] == row["true_tm2_code"])
            rec["pred_exact_sanskrit_match"] = row[f"top{k}_exact_sanskrit_match"]
            rec["query_token_overlap"] = row[f"top{k}_query_token_overlap"]
            rec["query_vs_pred_sans_overlap"] = row[f"top{k}_query_vs_pred_sans_overlap"]
            records.append(rec)
    return pd.DataFrame(records)

def derive_error_type(row):

    # safely read fields
    rerank_fixed = row.get("rerank_fixed", False)
    exact_match = row.get("top1_exact_sanskrit_match", False)
    token_overlap = row.get("token_overlap_ratio", 0.0)
    true_rank = row.get("true_rank", -1)

    # reranking fixed previous mistake
    if rerank_fixed:
        return "Rerank Fixed"

    # correct prediction
    if true_rank == 1:
        return "Correct Top1"

    # correct but not top1
    if 1 < true_rank <= 5:
        return "Correct But Lower Ranked"

    # lexical overlap but wrong
    if exact_match:
        return "Exact Sanskrit Confusion"

    # weak lexical overlap
    if token_overlap < 0.2:
        return "Low Lexical Overlap"

    # moderate ambiguity
    if token_overlap < 0.5:
        return "Semantic / Terminology Ambiguity"

    # fallback
    return "General Retrieval Error"

def build_pair_comparison(base_df: pd.DataFrame, rerank_df: pd.DataFrame, name: str) -> pd.DataFrame:
    comp = base_df[[
        "namc_code", "true_tm2_code", "query_text", "query_sanskrit_term",
        "top1_pred", "top1_score", "top1_correct", "true_rank"
    ]].copy()

    comp = comp.rename(columns={
        "top1_pred": f"{name}_base_top1_pred",
        "top1_score": f"{name}_base_top1_score",
        "top1_correct": f"{name}_base_top1_correct",
        "true_rank": f"{name}_base_true_rank",
    })

    comp[f"{name}_rerank_top1_pred"] = rerank_df["top1_pred"].values
    comp[f"{name}_rerank_top1_score"] = rerank_df["top1_score"].values
    comp[f"{name}_rerank_top1_correct"] = rerank_df["top1_correct"].values
    comp[f"{name}_rerank_true_rank"] = rerank_df["true_rank"].values
    comp[f"{name}_rerank_fixed"] = (~base_df["top1_correct"]) & (rerank_df["top1_correct"])
    comp[f"{name}_rerank_hurt"] = (base_df["top1_correct"]) & (~rerank_df["top1_correct"])
    return comp

def predict_from_scores(scores: np.ndarray, name: str, rerank: bool = False) -> pd.DataFrame:
    """
    Return a wide dataframe with top-K predictions + detailed fields for error analysis.
    """
    results = []
    tm2_lower = tm2["final_text_lower"].tolist()

    for i in tqdm(range(len(ayu_test)), desc=name):
        row = ayu_test.iloc[i]
        sims = np.array(scores[i], dtype=float)

        # normalize scores for stable ranking and thresholding
        sims = minmax_norm(sims)

        sans = row[A_SANS].lower().strip()
        q_text = row["final_query"]
        q_sans = row[A_SANS]
        true_code = row["true_tm2_code"]

        # lexical re-ranking using Sanskrit exact substring match
        if rerank and sans:
            for j in range(len(sims)):
                if sans in tm2_lower[j]:
                    sims[j] += RERANK_BOOST

        top_idx = np.argsort(sims)[-TOP_K:][::-1]
        ranked_codes = [tm2.iloc[j][T_CODE] for j in top_idx]

        # true rank in top-k list, if present
        true_rank = np.nan
        if true_code in ranked_codes:
            true_rank = ranked_codes.index(true_code) + 1

        rec = {
            "namc_code": row[A_NAMC],
            "true_tm2_code": true_code,
            "query_text": q_text,
            "query_sanskrit_term": q_sans,
            "algorithm": name,
            "top1_correct": ranked_codes[0] == true_code,
            "top5_hit": true_code in ranked_codes,
            "true_rank": true_rank,
            "rerank_applied": bool(rerank),
        }

        # Main top-K outputs
        for k, j in enumerate(top_idx, start=1):
            pred = tm2.iloc[j][T_CODE]
            score = float(sims[j])
            exact = bool(sans and sans in tm2_lower[j])
            tag = assign_tag(score, exact)
            cand = tm2.iloc[j]

            rec[f"top{k}_pred"] = pred
            rec[f"top{k}_score"] = score
            rec[f"top{k}_tag"] = tag
            rec[f"top{k}_title"] = cand[T_TITLE]
            rec[f"top{k}_index_terms"] = cand[T_INDEX]
            rec[f"top{k}_text"] = cand["final_text"]
            rec[f"top{k}_exact_sanskrit_match"] = exact
            rec[f"top{k}_query_token_overlap"] = compute_token_overlap(q_text, cand["final_text"])
            rec[f"top{k}_query_vs_pred_sans_overlap"] = compute_token_overlap(q_sans, cand["final_text"])

        # Helpful confidence style fields
        if len(top_idx) >= 2:
            rec["top1_minus_top2"] = float(sims[top_idx[0]] - sims[top_idx[1]])
        else:
            rec["top1_minus_top2"] = np.nan

        # true code score among all candidates (if present in TM2)
        if true_code in tm2_lookup.index:
            true_row = safe_get_tm2(true_code)
            true_row_id = int(true_row["tm2_row_id"])
            rec["true_candidate_row_id"] = true_row_id
            rec["true_candidate_title"] = true_row[T_TITLE]
            rec["true_candidate_index_terms"] = true_row[T_INDEX]
            rec["true_candidate_text"] = true_row["final_text"]
            rec["true_candidate_exact_sanskrit_match"] = bool(sans and sans in true_row["final_text_lower"])
            rec["true_candidate_token_overlap"] = compute_token_overlap(q_text, true_row["final_text"])
            rec["true_candidate_query_vs_sans_overlap"] = compute_token_overlap(q_sans, true_row["final_text"])
            rec["true_candidate_score"] = float(sims[true_row_id])
        else:
            rec["true_candidate_row_id"] = np.nan
            rec["true_candidate_title"] = ""
            rec["true_candidate_index_terms"] = ""
            rec["true_candidate_text"] = ""
            rec["true_candidate_exact_sanskrit_match"] = False
            rec["true_candidate_token_overlap"] = np.nan
            rec["true_candidate_query_vs_sans_overlap"] = np.nan
            rec["true_candidate_score"] = np.nan

        results.append(rec)

    df = pd.DataFrame(results)

    # rank-style helpers
    df["error_type"] = df.apply(derive_error_type, axis=1)

    # Confidence-style gap
    df["score_gap_to_true"] = df["top1_score"] - df["true_candidate_score"]

    return df

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
all_long_frames = []
error_frames = []
comparison_frames = []

for df in models:
    if df is None:
        continue

    algo = df["algorithm"].iloc[0]
    metrics = evaluate_predictions(df)
    print(f"\n🔹 {algo}")
    print(metrics)
    summary_rows.append({"algorithm": algo, **metrics})

    # Per-model error exports
    model_df = df.copy()
    model_df["true_code_in_top1"] = model_df["top1_correct"]
    model_df["true_code_in_top5"] = model_df["top5_hit"]
    model_df["error_type"] = model_df["error_type"].fillna("unknown")
    model_df["model"] = algo

    # Collect wrong or interesting cases only
    err_only = model_df.loc[~model_df["top1_correct"]].copy()
    if len(err_only) > 0:
        error_frames.append(err_only)

    all_long_frames.append(build_long_form(model_df, algo))

    # save a compact per-model error summary table
    model_error_summary_df = model_error_summary(model_df, algo)
    model_error_summary_df.to_csv(OUTPUT_DIR / f"error_summary_{algo.replace('+', '_').replace(' ', '_').lower()}.csv", index=False)

summary_df = pd.DataFrame(summary_rows)
print("\n=== SUMMARY TABLE ===")
print(summary_df)

# =========================================================
# PAIRWISE COMPARISON: BASE VS RERANK
# =========================================================
if pred_tfidf is not None and pred_tfidf_rerank is not None:
    comp_tfidf = build_pair_comparison(pred_tfidf, pred_tfidf_rerank, "tfidf")
    comparison_frames.append(comp_tfidf)
    comp_tfidf.to_csv(OUTPUT_DIR / "comparison_tfidf_base_vs_rerank.csv", index=False)

if pred_bm25 is not None and pred_bm25_rerank is not None:
    comp_bm25 = build_pair_comparison(pred_bm25, pred_bm25_rerank, "bm25")
    comparison_frames.append(comp_bm25)
    comp_bm25.to_csv(OUTPUT_DIR / "comparison_bm25_base_vs_rerank.csv", index=False)

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
# ERROR ANALYSIS EXPORTS
# =========================================================
if len(all_long_frames) > 0:
    long_all = pd.concat(all_long_frames, ignore_index=True)
    long_all.to_csv(OUTPUT_DIR / "all_models_long_topk.csv", index=False)
    print(f"\nSaved long-form top-K file: {OUTPUT_DIR / 'all_models_long_topk.csv'}")

if len(error_frames) > 0:
    all_errors = pd.concat(error_frames, ignore_index=True)
    # Order by most useful fields first
    preferred_cols = [
        "model", "algorithm", "namc_code", "query_text", "query_sanskrit_term",
        "true_tm2_code", "true_candidate_title", "true_candidate_score",
        "top1_pred", "top1_title", "top1_score", "top1_correct",
        "top5_hit", "true_rank", "top1_minus_top2", "score_gap_to_true",
        "error_type", "true_candidate_exact_sanskrit_match",
        "top1_exact_sanskrit_match", "top2_pred", "top3_pred", "top4_pred", "top5_pred"
    ]
    existing = [c for c in preferred_cols if c in all_errors.columns]
    remaining = [c for c in all_errors.columns if c not in existing]
    all_errors = all_errors[existing + remaining]
    all_errors.to_csv(OUTPUT_DIR / "all_top1_errors.csv", index=False)
    print(f"Saved error cases: {OUTPUT_DIR / 'all_top1_errors.csv'}")

summary_df.to_csv(OUTPUT_DIR / "comparison_summary.csv", index=False)

# =========================================================
# SAVE MODEL-LEVEL PREDICTIONS
# =========================================================
pred_tfidf.to_csv(OUTPUT_DIR / "predictions_tfidf.csv", index=False)
pred_tfidf_rerank.to_csv(OUTPUT_DIR / "predictions_tfidf_rerank.csv", index=False)
pred_bm25.to_csv(OUTPUT_DIR / "predictions_bm25.csv", index=False)
pred_bm25_rerank.to_csv(OUTPUT_DIR / "predictions_bm25_rerank.csv", index=False)

if pred_bert is not None:
    pred_bert.to_csv(OUTPUT_DIR / "predictions_bert.csv", index=False)
if pred_hybrid is not None:
    pred_hybrid.to_csv(OUTPUT_DIR / "predictions_hybrid.csv", index=False)

print("\n⏱️ Done")
print("Saved outputs to:", OUTPUT_DIR.resolve())
print("Key files:")
print(" - comparison_summary.csv")
print(" - all_models_long_topk.csv")
print(" - all_top1_errors.csv")
print(" - predictions_*.csv")
