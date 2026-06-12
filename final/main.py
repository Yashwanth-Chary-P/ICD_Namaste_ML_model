import pandas as pd
import unicodedata
import re


# -----------------------------------
# 1. NORMALIZATION (for search only)
# -----------------------------------
def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove (TM2)
    text = re.sub(r"\(tm2\)", "", text)

    # Unicode normalization (ś → s)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    # Remove special chars
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Clean spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------------
# 2. CLEAN INDEX TERMS (keep original)
# -----------------------------------
def extract_index_terms(text):
    if pd.isna(text):
        return []

    # Split by semicolon
    terms = [t.strip() for t in text.split(";") if t.strip()]

    cleaned = []
    for t in terms:
        # Remove (a), (b), (c)
        t = re.sub(r"\([a-z]\)", "", t)
        t = t.strip()

        if t:
            cleaned.append(t)

    return cleaned


# -----------------------------------
# 3. MAIN PREPROCESS FUNCTION
# -----------------------------------
def preprocess_tm2(csv_path):
    df = pd.read_csv(csv_path)

    processed_docs = []

    for _, row in df.iterrows():

        code = str(row.get("Code", "")).strip()
        title = str(row.get("title", "")).strip()

        # -------- ORIGINAL (for ML) --------
        original_title = title
        index_terms_original = extract_index_terms(row.get("Index Terms", ""))

        # -------- NORMALIZED (for search) --------
        normalized_title = normalize_text(title)
        index_terms_normalized = [normalize_text(t) for t in index_terms_original]

        # -------- BUILD ML TEXT --------
        ml_text = " ".join(
            [original_title] + index_terms_original
        )

        # -------- BUILD SEARCH TEXT --------
        search_text = " ".join(
            [normalized_title] + index_terms_normalized
        )

        # -------- OPTIONAL: KEYWORDS (top useful terms) --------
        keywords = index_terms_normalized[:20]  # limit for performance

        # -------- FINAL DOCUMENT --------
        doc = {
            "code": code,

            # Display
            "title": original_title,

            # Dual representation
            "original_title": original_title,
            "normalized_title": normalized_title,

            "index_terms_original": index_terms_original,
            "index_terms_normalized": index_terms_normalized,

            # Pipelines
            "ml_text": ml_text,
            "search_text": search_text,

            # Optional boost field
            "keywords": keywords,

            "source": "tm2"
        }

        processed_docs.append(doc)

    return processed_docs


# -----------------------------------
# 4. SAVE TO JSON
# -----------------------------------
def save_to_json(data, path="tm2_processed.json"):
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# -----------------------------------
# 5. RUN SCRIPT
# -----------------------------------
if __name__ == "__main__":
    data = preprocess_tm2("tm2.csv")
    save_to_json(data)

    print(f"✅ Processed {len(data)} TM2 records")