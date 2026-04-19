import requests
import csv
import time

# =========================
# CONFIG
# =========================
CLIENT_ID = "28fdd4a3-ac84-49b1-be00-50b0cacf78f7_8137f248-b335-4de3-8356-ead28e59e298"
CLIENT_SECRET = "j3SpHQKFE6tixUkywPCJcwBisIl/szKoCt5pUm9a54s="

TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"
BASE_URL = "https://id.who.int/icd/release/11/2026-01/mms"
ROOT_ID = "1435254666"

visited = set()
token = None

# =========================
# GET TOKEN (WITH RETRY)
# =========================
def get_token():
    global token

    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": "icdapi_access",
        "grant_type": "client_credentials"
    }

    for i in range(5):
        try:
            r = requests.post(TOKEN_URL, data=data, timeout=10)
            token = r.json()["access_token"]
            print("✅ Token fetched")
            return token
        except Exception:
            print(f"⚠️ Token retry {i+1}")

    raise Exception("❌ Failed to get token")

# =========================
# FETCH ENTITY (RETRY + REFRESH)
# =========================
def get_entity(entity_id):
    global token

    headers = {
        "Authorization": f"Bearer {token}",
        "API-Version": "v2",
        "Accept-Language": "en"
    }

    url = f"{BASE_URL}/{entity_id}"

    for i in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=10)

            # refresh token if expired
            if r.status_code == 401:
                print("🔄 Token expired, refreshing...")
                get_token()
                headers["Authorization"] = f"Bearer {token}"
                continue

            if r.status_code == 200:
                return r.json()

        except Exception:
            print(f"⚠️ Retry {i+1} for {entity_id}")

    return None

# =========================
# EXTRACT FIELDS
# =========================
def extract(data):
    code = data.get("code", "")

    if not code:
        return None

    title = data.get("title", {}).get("@value", "")
    fsn = data.get("fullySpecifiedName", {}).get("@value", "")
    desc = data.get("definition", {}).get("@value", "")

    inclusions = [
        inc["label"]["@value"]
        for inc in data.get("inclusion", [])
    ]

    exclusions = [
        exc["label"]["@value"]
        for exc in data.get("exclusion", [])
    ]

    return {
        "Code": code,
        "title": title,
        "Fully Specified Name": fsn,
        "Description": desc,
        "Inclusions": "; ".join(inclusions),
        "Exclusions": "; ".join(exclusions)
    }

# =========================
# DFS TRAVERSAL (WRITE LIVE)
# =========================
def dfs(entity_id, writer, file):
    if entity_id in visited:
        return

    visited.add(entity_id)

    data = get_entity(entity_id)
    if not data:
        return

    row = extract(data)

    if row:
        writer.writerow(row)
        file.flush()
        print("Saved:", row["Code"])

    for child in data.get("child", []):
        child_id = child.split("/")[-1]
        dfs(child_id, writer, file)

    time.sleep(0.05)

# =========================
# MAIN
# =========================
def main():
    get_token()

    with open("icd11.csv", "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=[
            "Code",
            "title",
            "Fully Specified Name",
            "Description",
            "Inclusions",
            "Exclusions"
        ])

        writer.writeheader()

        dfs(ROOT_ID, writer, f)

    print("✅ Done: icd11.csv")

if __name__ == "__main__":
    main()