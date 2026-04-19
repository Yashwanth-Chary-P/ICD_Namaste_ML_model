import requests
import csv
import time

CLIENT_ID = "28fdd4a3-ac84-49b1-be00-50b0cacf78f7_8137f248-b335-4de3-8356-ead28e59e298"
CLIENT_SECRET = "j3SpHQKFE6tixUkywPCJcwBisIl/szKoCt5pUm9a54s="

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
    token = r.json()["access_token"]

# =========================
# FETCH
# =========================
def get_entity(entity_id):
    global token

    url = f"{BASE_URL}/{entity_id}"

    for i in range(5):
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "API-Version": "v2",
                "Accept-Language": "en"
            }

            r = requests.get(url, headers=headers, timeout=10)

            if r.status_code == 401:
                get_token()
                continue

            if r.status_code == 200:
                return r.json()

        except:
            time.sleep(1)

    return None

# =========================
# EXTRACT
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
        "Inclusions": "; ".join([i["label"]["@value"] for i in data.get("inclusion", [])]),
        "Exclusions": "; ".join([e["label"]["@value"] for e in data.get("exclusion", [])])
    }

# =========================
# DFS
# =========================
def dfs(entity_id, writer, file):
    if entity_id in visited:
        return

    visited.add(entity_id)

    data = get_entity(entity_id)

    if data:
        row = extract(data)

        if row:
            writer.writerow(row)
            file.flush()
            print("Saved:", row["Code"])

        for child in data.get("child", []):
            child_id = child.split("/")[-1]
            dfs(child_id, writer, file)

    time.sleep(0.1)

# =========================
# MAIN (CHAPTER BASED)
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
            "Exclusions"
        ])

        writer.writeheader()

        dfs(chapter_id, writer, f)

    print(f"✅ Done: {filename}")


if __name__ == "__main__":
    get_token()

    # 👉 You will replace these IDs
    run_chapter("1630407678", "chapter2.csv")