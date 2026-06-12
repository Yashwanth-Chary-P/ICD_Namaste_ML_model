import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
os.makedirs("ablation_outputs", exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# CONFIG
# =========================================================

AYU_FILE = "eval_dataset_final.csv"
TM2_FILE = "tm2.csv"

TOP_K = 5
TEST_SIZE = 0.30
RANDOM_STATE = 42

OUTPUT_DIR = "ablation_outputs"

# =========================================================
# LOAD
# =========================================================

ayu = pd.read_csv(AYU_FILE)
tm2 = pd.read_csv(TM2_FILE)

ayu.columns = ayu.columns.str.lower().str.strip()
tm2.columns = tm2.columns.str.lower().str.strip()

# =========================================================
# COLUMN NAMES
# =========================================================

A_CODE = "namc_code"
A_TERM = "namc_term_diacritical"
A_SHORT = "short_definition"
A_LONG = "long_definition"
A_TM2 = "tm2_code"

T_CODE = "code"
T_TITLE = "title"
T_INDEX = "index terms"

# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = re.sub(r"\s+", " ", x)
    return x.strip()

for c in [A_TERM, A_SHORT, A_LONG]:
    ayu[c] = ayu[c].fillna("").apply(clean_text)

for c in [T_TITLE, T_INDEX]:
    tm2[c] = tm2[c].fillna("").apply(clean_text)

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

# =========================================================
# REMOVE RARE CLASSES
# =========================================================

counts = ayu[A_TM2].value_counts()

valid_classes = counts[counts >= 2].index

ayu_filtered = ayu[
    ayu[A_TM2].isin(valid_classes)
].copy()

print("\n📊 FILTERING")
print("Original rows:", len(ayu))
print("Filtered rows:", len(ayu_filtered))
print("Removed rare classes:", len(counts[counts < 2]))

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

train_df, test_df = train_test_split(
    ayu_filtered,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=ayu_filtered[A_TM2]
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("\n📊 SPLIT")
print("Train:", len(train_df))
print("Test :", len(test_df))

# =========================================================
# TM2 DOCUMENT BUILDERS
# =========================================================

def build_tm2_docs(
    repeat_index_terms=True,
    repeat_title=True
):

    docs = []

    for _, row in tm2.iterrows():

        title = row[T_TITLE]
        idx = row[T_INDEX]

        parts = []

        # title weighting
        if repeat_title:
            parts.extend([title] * 3)
        else:
            parts.append(title)

        # index term weighting
        if repeat_index_terms:
            parts.extend([idx] * 4)
        else:
            parts.append(idx)

        doc = " ".join(parts)

        docs.append(doc)

    return docs

# =========================================================
# QUERY BUILDERS
# =========================================================

def build_query(
    row,
    repeat_sanskrit=True,
    use_short=True,
    use_long=True
):

    parts = []

    sans = row[A_TERM]
    short = row[A_SHORT]
    longd = row[A_LONG]

    # Sanskrit repetition
    if repeat_sanskrit:
        parts.extend([sans] * 4)
    else:
        parts.append(sans)

    if use_short:
        parts.append(short)

    if use_long:
        parts.append(longd)

    return " ".join(parts)

# =========================================================
# RERANKING
# =========================================================

def rerank(
    scores,
    query,
    tm2_df,
    boost=0.25
):

    query_tokens = set(query.split())

    reranked = []

    for i, s in enumerate(scores):

        idx_terms = str(
            tm2_df.iloc[i][T_INDEX]
        ).lower()

        overlap = False

        for t in query_tokens:
            if t in idx_terms:
                overlap = True
                break

        if overlap:
            s = s + boost

        reranked.append(s)

    return np.array(reranked)

# =========================================================
# EVALUATION
# =========================================================

def evaluate(
    sim_matrix,
    test_df,
    rerank_enabled=False,
    rerank_boost=0.25
):

    top1 = 0
    top3 = 0
    top5 = 0

    rows = []

    for i in tqdm(range(len(test_df))):

        sims = sim_matrix[i].copy()

        query = test_queries[i]

        if rerank_enabled:
            sims = rerank(
                sims,
                query,
                tm2,
                boost=rerank_boost
            )

        order = np.argsort(sims)[::-1]

        topk = order[:TOP_K]

        pred_codes = [
            tm2.iloc[x][T_CODE]
            for x in topk
        ]

        true_code = test_df.iloc[i][A_TM2]

        if true_code in pred_codes[:1]:
            top1 += 1

        if true_code in pred_codes[:3]:
            top3 += 1

        if true_code in pred_codes[:5]:
            top5 += 1

        rows.append({
            "query": query,
            "true_tm2": true_code,
            "top1_pred": pred_codes[0],
            "top5_preds": pred_codes
        })

    n = len(test_df)

    metrics = {
        "Top1": top1 / n,
        "Top3": top3 / n,
        "Top5": top5 / n
    }

    return metrics, pd.DataFrame(rows)

# =========================================================
# ABLATION SETTINGS
# =========================================================

ABLATIONS = [

    # =========================================
    # FULL MODEL
    # =========================================

    {
        "name": "FULL_MODEL",
        "repeat_sanskrit": True,
        "repeat_index_terms": True,
        "repeat_title": True,
        "use_short": True,
        "use_long": True,
        "rerank": True,
        "boost": 0.25
    },

    # =========================================
    # REMOVE RERANKING
    # =========================================

    {
        "name": "NO_RERANK",
        "repeat_sanskrit": True,
        "repeat_index_terms": True,
        "repeat_title": True,
        "use_short": True,
        "use_long": True,
        "rerank": False,
        "boost": 0.0
    },

    # =========================================
    # REMOVE SANSKRIT REPETITION
    # =========================================

    {
        "name": "NO_SANSKRIT_REPEAT",
        "repeat_sanskrit": False,
        "repeat_index_terms": True,
        "repeat_title": True,
        "use_short": True,
        "use_long": True,
        "rerank": True,
        "boost": 0.25
    },

    # =========================================
    # REMOVE INDEX TERM WEIGHTING
    # =========================================

    {
        "name": "NO_INDEX_WEIGHTING",
        "repeat_sanskrit": True,
        "repeat_index_terms": False,
        "repeat_title": True,
        "use_short": True,
        "use_long": True,
        "rerank": True,
        "boost": 0.25
    },

    # =========================================
    # REMOVE LONG DEFINITION
    # =========================================

    {
        "name": "NO_LONG_DEFINITION",
        "repeat_sanskrit": True,
        "repeat_index_terms": True,
        "repeat_title": True,
        "use_short": True,
        "use_long": False,
        "rerank": True,
        "boost": 0.25
    },

    # =========================================
    # REMOVE SHORT DEFINITION
    # =========================================

    {
        "name": "NO_SHORT_DEFINITION",
        "repeat_sanskrit": True,
        "repeat_index_terms": True,
        "repeat_title": True,
        "use_short": False,
        "use_long": True,
        "rerank": True,
        "boost": 0.25
    },

    # =========================================
    # LOWER BOOST
    # =========================================

    {
        "name": "LOW_RERANK_BOOST",
        "repeat_sanskrit": True,
        "repeat_index_terms": True,
        "repeat_title": True,
        "use_short": True,
        "use_long": True,
        "rerank": True,
        "boost": 0.10
    },

    # =========================================
    # HIGH BOOST
    # =========================================

    {
        "name": "HIGH_RERANK_BOOST",
        "repeat_sanskrit": True,
        "repeat_index_terms": True,
        "repeat_title": True,
        "use_short": True,
        "use_long": True,
        "rerank": True,
        "boost": 0.50
    }

]

# =========================================================
# RUN ABLATIONS
# =========================================================

summary_rows = []

for cfg in ABLATIONS:

    print("\n" + "=" * 70)
    print("RUNNING:", cfg["name"])
    print("=" * 70)

    # build tm2 docs
    tm2_docs = build_tm2_docs(
        repeat_index_terms=cfg["repeat_index_terms"],
        repeat_title=cfg["repeat_title"]
    )

    # build queries
    global test_queries

    test_queries = [
        build_query(
            row,
            repeat_sanskrit=cfg["repeat_sanskrit"],
            use_short=cfg["use_short"],
            use_long=cfg["use_long"]
        )
        for _, row in test_df.iterrows()
    ]

    # tfidf
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    tm2_vecs = vectorizer.fit_transform(tm2_docs)

    query_vecs = vectorizer.transform(test_queries)

    sim = cosine_similarity(
        query_vecs,
        tm2_vecs
    )

    metrics, pred_df = evaluate(
        sim,
        test_df,
        rerank_enabled=cfg["rerank"],
        rerank_boost=cfg["boost"]
    )

    print(metrics)

    # save predictions
    pred_df.to_csv(
        f"{OUTPUT_DIR}/{cfg['name']}_predictions.csv",
        index=False
    )

    summary_rows.append({
        "ablation": cfg["name"],
        "Top1": metrics["Top1"],
        "Top3": metrics["Top3"],
        "Top5": metrics["Top5"],
        "repeat_sanskrit": cfg["repeat_sanskrit"],
        "repeat_index_terms": cfg["repeat_index_terms"],
        "repeat_title": cfg["repeat_title"],
        "use_short": cfg["use_short"],
        "use_long": cfg["use_long"],
        "rerank": cfg["rerank"],
        "boost": cfg["boost"]
    })

# =========================================================
# FINAL SUMMARY
# =========================================================

summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values(
    "Top1",
    ascending=False
)

print("\n📊 FINAL ABLATION RESULTS")
print(summary_df)

summary_df.to_csv(
    f"{OUTPUT_DIR}/ablation_summary.csv",
    index=False
)

print("\n✅ DONE")
print("Saved to:", OUTPUT_DIR)

print("\nIMPORTANT FILES TO SEND:")
print("1. ablation_summary.csv")
print("2. FULL_MODEL_predictions.csv")
print("3. NO_RERANK_predictions.csv")
print("4. NO_SANSKRIT_REPEAT_predictions.csv")