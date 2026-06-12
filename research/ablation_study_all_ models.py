import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rank_bm25 import BM25Okapi

# =========================================================
# OPTIONAL BERT
# =========================================================

try:
    from sentence_transformers import SentenceTransformer, util
    HAVE_BERT = True
except Exception:
    HAVE_BERT = False

# =========================================================
# CONFIG
# =========================================================

AYU_FILE = "eval_dataset_final.csv"
TM2_FILE = "tm2.csv"

OUTPUT_DIR = "complete_model_ablation"

TOP_K = 5
TEST_SIZE = 0.30
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# REMOVE RARE CLASSES
# =========================================================

counts = ayu[A_TM2].value_counts()

valid_classes = counts[counts >= 2].index

ayu = ayu[
    ayu[A_TM2].isin(valid_classes)
].copy()

print("\n📊 FILTERING")
print("Remaining rows:", len(ayu))

# =========================================================
# SPLIT
# =========================================================

train_df, test_df = train_test_split(
    ayu,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=ayu[A_TM2]
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("\n📊 SPLIT")
print("Train:", len(train_df))
print("Test :", len(test_df))

# =========================================================
# QUERY BUILDER
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
# TM2 DOC BUILDER
# =========================================================

def build_tm2_docs(
    repeat_title=True,
    repeat_index=True
):

    docs = []

    for _, row in tm2.iterrows():

        title = row[T_TITLE]
        idx = row[T_INDEX]

        parts = []

        if repeat_title:
            parts.extend([title] * 3)
        else:
            parts.append(title)

        if repeat_index:
            parts.extend([idx] * 4)
        else:
            parts.append(idx)

        docs.append(" ".join(parts))

    return docs

# =========================================================
# RERANK
# =========================================================

def rerank(
    scores,
    query,
    tm2_docs,
    boost=0.25
):

    q_tokens = set(query.split())

    new_scores = []

    for i, s in enumerate(scores):

        doc = tm2_docs[i]

        overlap = False

        for t in q_tokens:

            if t in doc:
                overlap = True
                break

        if overlap:
            s += boost

        new_scores.append(s)

    return np.array(new_scores)

# =========================================================
# METRIC EVAL
# =========================================================

def evaluate_predictions(
    pred_lists,
    true_codes
):

    top1 = 0
    top3 = 0
    top5 = 0

    for preds, true_code in zip(pred_lists, true_codes):

        if true_code in preds[:1]:
            top1 += 1

        if true_code in preds[:3]:
            top3 += 1

        if true_code in preds[:5]:
            top5 += 1

    n = len(true_codes)

    return {
        "Top1": top1 / n,
        "Top3": top3 / n,
        "Top5": top5 / n
    }

# =========================================================
# EXPERIMENT CONFIGS
# =========================================================

EXPERIMENTS = [

    # =====================================================
    # TFIDF
    # =====================================================

    {
        "name": "TFIDF_FULL",
        "model": "tfidf",
        "repeat_sanskrit": True,
        "repeat_index": True,
        "use_long": True,
        "rerank": True
    },

    {
        "name": "TFIDF_NO_SANSKRIT",
        "model": "tfidf",
        "repeat_sanskrit": False,
        "repeat_index": True,
        "use_long": True,
        "rerank": True
    },

    {
        "name": "TFIDF_NO_INDEX",
        "model": "tfidf",
        "repeat_sanskrit": True,
        "repeat_index": False,
        "use_long": True,
        "rerank": True
    },

    {
        "name": "TFIDF_NO_LONG",
        "model": "tfidf",
        "repeat_sanskrit": True,
        "repeat_index": True,
        "use_long": False,
        "rerank": True
    },

    {
        "name": "TFIDF_NO_RERANK",
        "model": "tfidf",
        "repeat_sanskrit": True,
        "repeat_index": True,
        "use_long": True,
        "rerank": False
    },

    # =====================================================
    # BM25
    # =====================================================

    {
        "name": "BM25_FULL",
        "model": "bm25",
        "repeat_sanskrit": True,
        "repeat_index": True,
        "use_long": True,
        "rerank": True
    },

    {
        "name": "BM25_NO_SANSKRIT",
        "model": "bm25",
        "repeat_sanskrit": False,
        "repeat_index": True,
        "use_long": True,
        "rerank": True
    },

    {
        "name": "BM25_NO_INDEX",
        "model": "bm25",
        "repeat_sanskrit": True,
        "repeat_index": False,
        "use_long": True,
        "rerank": True
    },

    {
        "name": "BM25_NO_LONG",
        "model": "bm25",
        "repeat_sanskrit": True,
        "repeat_index": True,
        "use_long": False,
        "rerank": True
    },

    {
        "name": "BM25_NO_RERANK",
        "model": "bm25",
        "repeat_sanskrit": True,
        "repeat_index": True,
        "use_long": True,
        "rerank": False
    }

]

# =========================================================
# OPTIONAL BERT
# =========================================================

if HAVE_BERT:

    EXPERIMENTS.extend([

        {
            "name": "BERT_FULL",
            "model": "bert",
            "repeat_sanskrit": True,
            "repeat_index": True,
            "use_long": True,
            "rerank": False
        },

        {
            "name": "BERT_NO_LONG",
            "model": "bert",
            "repeat_sanskrit": True,
            "repeat_index": True,
            "use_long": False,
            "rerank": False
        }

    ])

# =========================================================
# MAIN LOOP
# =========================================================

summary = []

if HAVE_BERT:
    print("\n🧠 Loading BERT...")
    bert_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

for cfg in EXPERIMENTS:

    print("\n" + "=" * 70)
    print("RUNNING:", cfg["name"])
    print("=" * 70)

    # =====================================================
    # BUILD DOCS
    # =====================================================

    tm2_docs = build_tm2_docs(
        repeat_title=True,
        repeat_index=cfg["repeat_index"]
    )

    test_queries = [

        build_query(
            row,
            repeat_sanskrit=cfg["repeat_sanskrit"],
            use_short=True,
            use_long=cfg["use_long"]
        )

        for _, row in test_df.iterrows()
    ]

    # =====================================================
    # TFIDF
    # =====================================================

    if cfg["model"] == "tfidf":

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2)
        )

        tm2_vecs = vectorizer.fit_transform(tm2_docs)

        query_vecs = vectorizer.transform(test_queries)

        sim_matrix = cosine_similarity(
            query_vecs,
            tm2_vecs
        )

    # =====================================================
    # BM25
    # =====================================================

    elif cfg["model"] == "bm25":

        tokenized_docs = [
            d.split()
            for d in tm2_docs
        ]

        bm25 = BM25Okapi(tokenized_docs)

        sim_matrix = []

        for q in test_queries:

            scores = bm25.get_scores(q.split())

            sim_matrix.append(scores)

        sim_matrix = np.array(sim_matrix)

    # =====================================================
    # BERT
    # =====================================================

    elif cfg["model"] == "bert":

        tm2_emb = bert_model.encode(
            tm2_docs,
            convert_to_tensor=True
        )

        q_emb = bert_model.encode(
            test_queries,
            convert_to_tensor=True
        )

        sim_matrix = util.cos_sim(
            q_emb,
            tm2_emb
        ).cpu().numpy()

    # =====================================================
    # PREDICTIONS
    # =====================================================

    pred_lists = []

    for i in tqdm(range(len(test_df))):

        sims = sim_matrix[i].copy()

        if cfg["rerank"]:

            sims = rerank(
                sims,
                test_queries[i],
                tm2_docs
            )

        order = np.argsort(sims)[::-1]

        preds = [
            tm2.iloc[x][T_CODE]
            for x in order[:TOP_K]
        ]

        pred_lists.append(preds)

    # =====================================================
    # METRICS
    # =====================================================

    metrics = evaluate_predictions(
        pred_lists,
        test_df[A_TM2].tolist()
    )

    print(metrics)

    # =====================================================
    # SAVE PREDICTIONS
    # =====================================================

    pred_df = pd.DataFrame({
        "query": test_queries,
        "true_tm2": test_df[A_TM2].tolist(),
        "top5_preds": pred_lists
    })

    pred_df.to_csv(
        f"{OUTPUT_DIR}/{cfg['name']}_predictions.csv",
        index=False
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    summary.append({

        "experiment": cfg["name"],
        "model": cfg["model"],

        "Top1": metrics["Top1"],
        "Top3": metrics["Top3"],
        "Top5": metrics["Top5"],

        "repeat_sanskrit":
            cfg["repeat_sanskrit"],

        "repeat_index":
            cfg["repeat_index"],

        "use_long":
            cfg["use_long"],

        "rerank":
            cfg["rerank"]
    })

# =========================================================
# FINAL SAVE
# =========================================================

summary_df = pd.DataFrame(summary)

summary_df = summary_df.sort_values(
    "Top1",
    ascending=False
)

print("\n📊 FINAL SUMMARY")
print(summary_df)

summary_df.to_csv(
    f"{OUTPUT_DIR}/complete_ablation_summary.csv",
    index=False
)

print("\n✅ DONE")
print("Saved to:", OUTPUT_DIR)

print("\nIMPORTANT OUTPUT:")
print("complete_ablation_summary.csv")