import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# INPUT FILES
# =========================================================
AYURVEDA_FILE = "AYURVEDA_processed.csv"
ICD_FILE = "tm2_final.csv"

OUTPUT_FILE = "mapping_output.csv"

# =========================================================
# LOAD DATA
# =========================================================
print("📥 Loading datasets...")

ayu = pd.read_csv(AYURVEDA_FILE)
icd = pd.read_csv(ICD_FILE)

print("Ayurveda:", ayu.shape)
print("ICD:", icd.shape)

# Fill missing safely
ayu["query"] = ayu["query"].fillna("")
icd["index_terms"] = icd["index_terms"].fillna("")

# =========================================================
# TF-IDF VECTORIZATION
# =========================================================
print("\n🧠 Building TF-IDF model...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2),   # unigrams + bigrams
    stop_words="english"
)

# Fit on ICD (search space)
icd_vectors = vectorizer.fit_transform(icd["index_terms"])

# Transform Ayurveda queries
ayu_vectors = vectorizer.transform(ayu["query"])

print("ICD vector shape:", icd_vectors.shape)
print("Ayurveda vector shape:", ayu_vectors.shape)

# =========================================================
# SIMILARITY COMPUTATION
# =========================================================
print("\n🔍 Computing similarity...")

similarity_matrix = cosine_similarity(ayu_vectors, icd_vectors)

# =========================================================
# FIND BEST MATCH
# =========================================================
print("\n📊 Finding best matches...")

results = []

for i in range(similarity_matrix.shape[0]):

    best_idx = similarity_matrix[i].argmax()
    best_score = similarity_matrix[i][best_idx]

    ayu_term = ayu.iloc[i]["namc_term_diacritical"]
    icd_code = icd.iloc[best_idx]["code"]
    icd_title = icd.iloc[best_idx]["title"]

    # Classification
    if best_score >= 0.85:
        relation = "equivalent"
    elif best_score >= 0.60:
        relation = "narrower"
    elif best_score >= 0.40:
        relation = "related"
    else:
        relation = "weak"

    results.append({
        "ayurveda_term": ayu_term,
        "icd_code": icd_code,
        "icd_title": icd_title,
        "similarity_score": round(float(best_score), 4),
        "relationship": relation
    })

# =========================================================
# SAVE OUTPUT
# =========================================================
result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ Mapping completed!")
print("💾 Saved:", OUTPUT_FILE)

print("\n🔍 Sample:")
print(result_df.head(10))