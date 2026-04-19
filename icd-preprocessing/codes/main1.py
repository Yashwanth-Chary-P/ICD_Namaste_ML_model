import requests
import csv
import time

# =========================
# CONFIG
# =========================
CLIENT_ID = "28fdd4a3-ac84-49b1-be00-50b0cacf78f7_98caff9b-d075-4984-9454-f4eed404029f"
CLIENT_SECRET = "XkMZWt4GIQlQr6pTkbZPwGDRVZ5HOuyRFZurEgBRvps="

TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"
BASE_URL = "https://id.who.int/icd/release/11/2026-01/mms"

visited = set()
token = None


# =========================
# TOKEN
# =========================
def get_token():
    global token

    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": "icdapi_access",
        "grant_type": "client_credentials"
    }

    r = requests.post(TOKEN_URL, data=data, timeout=10)
    token = r.json().get("access_token")

    print("🔑 Token refreshed")


# =========================
# FETCH ENTITY (NORMAL + RESIDUAL)
# =========================
def get_entity(entity_id, residual=None):
    global token

    if residual:
        url = f"{BASE_URL}/{entity_id}/{residual}"
    else:
        url = f"{BASE_URL}/{entity_id}"

    for _ in range(5):
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "API-Version": "v2",
                "Accept-Language": "en"
            }

            r = requests.get(url, headers=headers, timeout=10)

            if r.status_code == 401:
                print("⚠️ Token expired. Refreshing...")
                get_token()
                continue

            if r.status_code == 200:
                return r.json()

        except Exception as e:
            time.sleep(1)

    return None


# =========================
# SAFE JOIN
# =========================
def safe_join(items):
    return "; ".join([
        i.get("label", {}).get("@value", "")
        for i in items
        if i.get("label")
    ])


# =========================
# EXTRACT DATA
# =========================
def extract(data):
    code = data.get("code", "")
    if not code:
        return None

    return {
        "Code": code,
        "title": data.get("title", {}).get("@value", ""),
        "Fully Specified Name": data.get("fullySpecifiedName", {}).get("@value", ""),
        "Description": data.get("definition", {}).get("@value", ""),
        "Inclusions": safe_join(data.get("inclusion", [])),
        "Exclusions": safe_join(data.get("exclusion", [])),
        "Index Terms": safe_join(data.get("indexTerm", []))
    }


# =========================
# DFS TRAVERSAL
# =========================
def dfs(entity_id, writer, file):
    if entity_id in visited:
        return

    visited.add(entity_id)

    # =========================
    # NORMAL ENTITY
    # =========================
    data = get_entity(entity_id)

    if data:
        row = extract(data)

        if row:
            writer.writerow(row)
            file.flush()
            print("✅ Saved:", row["Code"])

        # =========================
        # 🔴 FETCH RESIDUAL NODES
        # =========================
        for residual in ["other", "unspecified"]:
            residual_data = get_entity(entity_id, residual)

            if residual_data:
                row = extract(residual_data)

                if row:
                    writer.writerow(row)
                    file.flush()
                    print("🔴 Saved Residual:", row["Code"])

        # =========================
        # CHILD TRAVERSAL
        # =========================
        for child in data.get("child", []):
            child_id = child.split("/")[-1]
            dfs(child_id, writer, file)

    time.sleep(0.05)  # faster but safe


# =========================
# RUN CHAPTER
# =========================
def run_chapter(chapter_id, filename):
    global visited
    visited = set()

    with open(filename, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=[
            "Code",
            "title",
            "Fully Specified Name",
            "Description",
            "Inclusions",
            "Exclusions",
            "Index Terms"
        ])

        writer.writeheader()

        dfs(chapter_id, writer, f)

    print(f"\n🎉 DONE: {filename}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    get_token()

    # Example: Chapter 18 (TM2 included here)
    run_chapter(562274788, "tm2.csv")