import os
import re
import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
ERRORS_FILE = "outputs_error_analysis/all_top1_errors.csv"
TM2_FILE = "tm2.csv"
OUT_DIR = "qualitative_example_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# How many examples to keep in each group
N_RECOVERABLE = 4      # true_rank 2..5  -> ranking failure
N_RECALL_FAIL = 4      # true_rank >5 or missing -> retrieval failure
N_ATTRACTOR = 4        # top1_pred is a false-positive hub
N_AMBIGUOUS = 4        # same query maps to many different predictions

# =========================================================
# LOAD
# =========================================================
errors = pd.read_csv(ERRORS_FILE)
tm2 = pd.read_csv(TM2_FILE)

# Normalize column names
errors.columns = [c.strip().lower() for c in errors.columns]
tm2.columns = [c.strip().lower().replace(" ", "_") for c in tm2.columns]

# =========================================================
# HELPERS
# =========================================================
def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of these columns found: {candidates}")
    return None

def safe_num(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)

def short_text(x, max_len=160):
    s = safe_str(x).replace("\n", " ").strip()
    return s if len(s) <= max_len else s[:max_len - 3] + "..."

def make_topk_brief(row, prefix="top"):
    bits = []
    for k in range(1, 6):
        pred = safe_str(row.get(f"{prefix}{k}_pred", ""))
        score = row.get(f"{prefix}{k}_score", np.nan)
        title = safe_str(row.get(f"{prefix}{k}_title", ""))
        if pred:
            if title and title != pred:
                bits.append(f"{k}:{pred} | {title} ({score:.3f})")
            else:
                bits.append(f"{k}:{pred} ({score:.3f})")
    return " || ".join(bits)

def infer_reason(row, hub_set):
    model = safe_str(row.get("model", "")).upper()
    true_rank = safe_num(row.get("true_rank", np.nan))
    top1_pred = safe_str(row.get("top1_pred", ""))
    top1_overlap = safe_num(row.get("top1_query_token_overlap", np.nan))
    score_gap = safe_num(row.get("score_gap_to_true", np.nan))
    error_type = safe_str(row.get("error_type", ""))

    if top1_pred in hub_set:
        return "Attractor concept: this TM2 code repeatedly absorbs unrelated queries."

    if error_type == "Exact Sanskrit Confusion":
        return "Exact Sanskrit/alias confusion: the system confused closely related transliterations or synonyms."

    if pd.notna(true_rank) and 1 < true_rank <= 5:
        if pd.notna(score_gap) and score_gap < 0.15:
            return "Near-miss ranking failure: the correct code is retrieved, but a slightly stronger attractor wins."
        return "Ranking failure: the correct code is inside Top-5, so recall is fine but ordering is imperfect."

    if pd.isna(true_rank) or true_rank > 5:
        if model == "BERT":
            return "Dense retrieval failure: the correct code was not surfaced in Top-5, suggesting semantic drift / weak domain grounding."
        if pd.notna(top1_overlap) and top1_overlap < 0.20:
            return "Sparse lexical failure: the query and candidates share too little surface evidence."
        return "Retrieval miss: the correct TM2 code is absent from Top-5, so this is a recall failure."

    return "General retrieval error."

# Column mapping
model_col = pick_col(errors, ["model"])
query_col = pick_col(errors, ["query_sanskrit_term", "query_term", "query_text"])
true_code_col = pick_col(errors, ["true_tm2_code", "true_code", "tm2_code"])
true_title_col = pick_col(errors, ["true_candidate_title", "true_title"], required=False)
top1_pred_col = pick_col(errors, ["top1_pred"])
top1_title_col = pick_col(errors, ["top1_title"], required=False)
true_rank_col = pick_col(errors, ["true_rank"])
error_type_col = pick_col(errors, ["error_type"], required=False)
score_gap_col = pick_col(errors, ["score_gap_to_true"], required=False)
top1_overlap_col = pick_col(errors, ["top1_query_token_overlap"], required=False)

# =========================================================
# HUB / ATTRACTOR CODES
# =========================================================
hub_counts = errors[top1_pred_col].value_counts()
hub_codes = set(hub_counts.head(10).index.tolist())

hub_df = pd.DataFrame({
    "tm2_code": hub_counts.head(10).index,
    "false_positive_count": hub_counts.head(10).values
})

# attach titles if possible
tm2_code_col = pick_col(tm2, ["code", "tm2_code"])
tm2_title_col = pick_col(tm2, ["title", "tm2_title"], required=False)

if tm2_code_col in tm2.columns and tm2_title_col:
    hub_df = hub_df.merge(
        tm2[[tm2_code_col, tm2_title_col]].drop_duplicates(),
        left_on="tm2_code",
        right_on=tm2_code_col,
        how="left"
    )

hub_df.to_csv(os.path.join(OUT_DIR, "top_false_positive_hubs.csv"), index=False)

# =========================================================
# QUERY AMBIGUITY TABLE
# =========================================================
ambig = (
    errors.groupby(query_col)
    .agg(
        num_unique_predictions=(top1_pred_col, "nunique"),
        num_models=(model_col, "nunique"),
        error_count=(query_col, "size")
    )
    .reset_index()
    .rename(columns={query_col: "query_sanskrit_term"})
)

ambig = ambig.sort_values(
    ["num_unique_predictions", "error_count", "num_models"],
    ascending=[False, False, False]
)

ambig.to_csv(os.path.join(OUT_DIR, "query_ambiguity_all.csv"), index=False)

# =========================================================
# SELECT EXAMPLES
# =========================================================
base = errors.copy()

# standardize helper columns
base["__true_rank"] = base[true_rank_col].apply(safe_num)
base["__score_gap"] = base[score_gap_col].apply(safe_num) if score_gap_col else np.nan
base["__top1_overlap"] = base[top1_overlap_col].apply(safe_num) if top1_overlap_col else np.nan
base["__topk"] = base.apply(make_topk_brief, axis=1)
base["__reason"] = base.apply(lambda r: infer_reason(r, hub_codes), axis=1)

# prefer rows that are informative and not duplicates
dedup_keys = [query_col, true_code_col, top1_pred_col, model_col]

# 1) Recoverable ranking failures: true rank 2..5
recoverable = base[
    base["__true_rank"].between(2, 5, inclusive="both")
].copy()

recoverable = recoverable.sort_values(
    by=["__true_rank", "__score_gap", "__top1_overlap"],
    ascending=[True, True, True]
).drop_duplicates(subset=dedup_keys)

recoverable = recoverable.head(N_RECOVERABLE)
recoverable["category"] = "Recoverable ranking failure"

# 2) Recall failures: true rank missing or >5
recall_fail = base[
    base["__true_rank"].isna() | (base["__true_rank"] > 5)
].copy()

# prioritize BERT first, then HYBRID, then BM25, then TFIDF
model_priority = {"BERT": 0, "HYBRID": 1, "BM25": 2, "BM25+RERANK": 3, "TFIDF": 4, "TFIDF+RERANK": 5}
recall_fail["__model_priority"] = recall_fail[model_col].map(lambda x: model_priority.get(safe_str(x).upper(), 99))

recall_fail = recall_fail.sort_values(
    by=["__model_priority", "__top1_overlap", "__score_gap"],
    ascending=[True, True, True]
).drop_duplicates(subset=dedup_keys)

recall_fail = recall_fail.head(N_RECALL_FAIL)
recall_fail["category"] = "Recall failure"

# 3) Attractor errors: top1 prediction is one of the hub codes
attractor = base[base[top1_pred_col].isin(hub_codes)].copy()
attractor["hub_fp_count"] = attractor[top1_pred_col].map(hub_counts.to_dict())
attractor = attractor.sort_values(
    by=["hub_fp_count", "__score_gap", "__top1_overlap"],
    ascending=[False, True, True]
).drop_duplicates(subset=dedup_keys)

attractor = attractor.head(N_ATTRACTOR)
attractor["category"] = "Semantic attractor"

# 4) Ambiguous queries: same Sanskrit query maps to many distinct predictions
ambig_counts = (
    base.groupby(query_col)[top1_pred_col]
    .nunique()
    .reset_index()
    .rename(columns={top1_pred_col: "num_unique_predictions"})
)

ambig_counts = ambig_counts.sort_values(
    "num_unique_predictions", ascending=False
)

ambig_queries = ambig_counts[ambig_counts["num_unique_predictions"] >= 2].head(N_AMBIGUOUS)

ambig_rows = []
for q in ambig_queries[query_col].tolist():
    sub = base[base[query_col] == q].copy()
    sub["num_unique_predictions"] = sub[top1_pred_col].nunique()
    # choose the most informative row for that query
    sub = sub.sort_values(
        by=["num_unique_predictions", "__true_rank", "__score_gap", "__top1_overlap"],
        ascending=[False, True, True, True]
    )
    row = sub.iloc[0].copy()
    row["category"] = "Query ambiguity"
    row["query_prediction_set"] = " | ".join(sorted(sub[top1_pred_col].astype(str).unique().tolist()))
    row["query_model_set"] = " | ".join(sorted(sub[model_col].astype(str).unique().tolist()))
    row["query_unique_prediction_count"] = int(sub[top1_pred_col].nunique())
    row["__reason"] = (
        "Same Sanskrit query produces multiple distinct TM2 predictions across models, "
        "which indicates ontology-level ambiguity rather than a single isolated ranking mistake."
    )
    ambig_rows.append(row)

ambiguous = pd.DataFrame(ambig_rows)

# =========================================================
# FINAL CURATED TABLE
# =========================================================
def tidy_output(df):
    if df.empty:
        return df
    cols = [
        "category",
        model_col,
        query_col,
        true_code_col,
        true_title_col if true_title_col else None,
        top1_pred_col,
        top1_title_col if top1_title_col else None,
        true_rank_col,
        error_type_col if error_type_col else None,
        "__top1_overlap",
        "__score_gap",
        "__topk",
        "__reason",
    ]
    cols = [c for c in cols if c is not None and c in df.columns]
    extra = [c for c in ["hub_fp_count", "query_unique_prediction_count", "query_prediction_set", "query_model_set"] if c in df.columns]
    cols = cols + extra
    out = df[cols].copy()
    # rename for readability
    rename_map = {
        model_col: "model",
        query_col: "query_sanskrit_term",
        true_code_col: "true_tm2_code",
        top1_pred_col: "top1_pred",
        true_rank_col: "true_rank",
        "__top1_overlap": "top1_query_token_overlap",
        "__score_gap": "score_gap_to_true",
        "__topk": "top5_brief",
        "__reason": "why_this_example",
    }
    if true_title_col:
        rename_map[true_title_col] = "true_candidate_title"
    if top1_title_col:
        rename_map[top1_title_col] = "top1_title"
    if error_type_col:
        rename_map[error_type_col] = "error_type"
    out = out.rename(columns=rename_map)
    return out

selected = pd.concat(
    [recoverable, recall_fail, attractor],
    ignore_index=True
)

selected = tidy_output(selected)
ambiguous_out = ambiguous.copy()
if not ambiguous_out.empty:
    cols = [c for c in [
        "category",
        model_col,
        query_col,
        true_code_col,
        true_title_col if true_title_col else None,
        top1_pred_col,
        top1_title_col if top1_title_col else None,
        true_rank_col,
        error_type_col if error_type_col else None,
        "__top1_overlap",
        "__score_gap",
        "__reason",
        "query_unique_prediction_count",
        "query_prediction_set",
        "query_model_set",
    ] if c is not None and c in ambiguous_out.columns]
    ambiguous_out = ambiguous_out[cols].copy()
    ambiguous_out = ambiguous_out.rename(columns={
        model_col: "model",
        query_col: "query_sanskrit_term",
        true_code_col: "true_tm2_code",
        top1_pred_col: "top1_pred",
        true_rank_col: "true_rank",
        "__top1_overlap": "top1_query_token_overlap",
        "__score_gap": "score_gap_to_true",
        "__reason": "why_this_example",
    })
    if true_title_col:
        ambiguous_out = ambiguous_out.rename(columns={true_title_col: "true_candidate_title"})
    if top1_title_col:
        ambiguous_out = ambiguous_out.rename(columns={top1_title_col: "top1_title"})
    if error_type_col:
        ambiguous_out = ambiguous_out.rename(columns={error_type_col: "error_type"})

# =========================================================
# ADD SHORT HUMAN-READABLE EXPLANATIONS
# =========================================================
def summarize_row(row):
    q = short_text(row.get("query_sanskrit_term", ""), 90)
    gt = short_text(row.get("true_candidate_title", row.get("true_tm2_code", "")), 60)
    pred = short_text(row.get("top1_title", row.get("top1_pred", "")), 60)
    cat = short_text(row.get("category", ""), 40)
    reason = short_text(row.get("why_this_example", ""), 160)
    return f"[{cat}] {q} -> GT: {gt} | Pred: {pred} | {reason}"

if not selected.empty:
    selected["quick_note"] = selected.apply(summarize_row, axis=1)

if not ambiguous_out.empty:
    ambiguous_out["quick_note"] = ambiguous_out.apply(summarize_row, axis=1)

# =========================================================
# SAVE OUTPUTS
# =========================================================
selected_path = os.path.join(OUT_DIR, "qualitative_examples_selected.csv")
ambiguous_path = os.path.join(OUT_DIR, "qualitative_examples_ambiguous.csv")
summary_path = os.path.join(OUT_DIR, "qualitative_examples_summary.csv")

selected.to_csv(selected_path, index=False)
ambiguous_out.to_csv(ambiguous_path, index=False)

summary = pd.DataFrame([
    {"file": "qualitative_examples_selected.csv", "rows": len(selected)},
    {"file": "qualitative_examples_ambiguous.csv", "rows": len(ambiguous_out)},
    {"file": "top_false_positive_hubs.csv", "rows": len(hub_df)},
    {"file": "query_ambiguity_all.csv", "rows": len(ambig)},
])

summary.to_csv(summary_path, index=False)

# =========================================================
# PRINT
# =========================================================
print("\n✅ Saved files to:", OUT_DIR)
print("\nSelected examples:")
print(selected[["category", "model", "query_sanskrit_term", "true_tm2_code", "top1_pred", "true_rank", "why_this_example"]].to_string(index=False))

print("\nAmbiguous examples:")
if not ambiguous_out.empty:
    print(ambiguous_out[["category", "model", "query_sanskrit_term", "true_tm2_code", "top1_pred", "query_unique_prediction_count", "why_this_example"]].to_string(index=False))
else:
    print("No ambiguous examples selected.")

print("\nSend me these files:")
print("1) qualitative_example_outputs/qualitative_examples_selected.csv")
print("2) qualitative_example_outputs/qualitative_examples_ambiguous.csv")
print("3) qualitative_example_outputs/qualitative_examples_summary.csv")