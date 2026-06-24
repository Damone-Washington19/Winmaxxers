import json
import re
import time
import random
import requests
from pathlib import Path

API_URL = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "WinmaxxersBot/1.0 (https://example.com; contact@example.com)",
    "Accept": "application/json"
}

MAX_RETRIES = 10
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

MIN_REQUEST_DELAY = 1.0
JITTER = 0.5

SCRIPT_DIR = Path(__file__).resolve().parent
MASTER_PAGES_FILE = SCRIPT_DIR / "master_pages.json"

WIKILINK_PATTERN = re.compile(r"\[\[([^\[\]|#]+)")


# -----------------------------
# Request helper (robust)
# -----------------------------
def request_with_retries(params):
    delay = 1.0

    for attempt in range(1, MAX_RETRIES + 1):

        response = requests.get(
            API_URL,
            params=params,
            headers=HEADERS,
            timeout=30
        )

        if response.status_code not in RETRY_STATUS_CODES:
            response.raise_for_status()
            return response

        if attempt == MAX_RETRIES:
            response.raise_for_status()

        time.sleep(delay)
        delay *= 2

    raise RuntimeError("Exceeded retry limit")


# -----------------------------
# Wikipedia helpers
# -----------------------------
def get_revision_before(title: str, timestamp: str):
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": title,
        "rvlimit": 1,
        "rvdir": "older",
        "rvstart": timestamp,
        "rvprop": "ids|timestamp"
    }

    response = request_with_retries(params)
    data = response.json()

    page = next(iter(data["query"]["pages"].values()))
    revisions = page.get("revisions", [])

    return revisions[0] if revisions else None


def get_revision_wikitext(revid: int):
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "revids": revid,
        "rvprop": "content",
        "rvslots": "main"
    }

    response = request_with_retries(params)
    data = response.json()

    page = next(iter(data["query"]["pages"].values()))
    return page["revisions"][0]["slots"]["main"]["*"]


def get_categories(title: str):
    categories = []
    clcontinue = None

    while True:
        params = {
            "action": "query",
            "format": "json",
            "prop": "categories",
            "titles": title,
            "cllimit": "max"
        }

        if clcontinue:
            params["clcontinue"] = clcontinue

        response = request_with_retries(params)
        data = response.json()

        page = next(iter(data["query"]["pages"].values()))

        categories.extend(
            cat["title"] for cat in page.get("categories", [])
        )

        if "continue" not in data:
            break

        clcontinue = data["continue"]["clcontinue"]

    return categories


def extract_links(wikitext: str):
    links = set()

    for match in WIKILINK_PATTERN.findall(wikitext):
        title = match.strip()

        if not title:
            continue

        if ":" in title:
            continue

        links.add(title)

    return sorted(links)


# -----------------------------
# Persistence
# -----------------------------
def load_snapshot(year: int):
    path = Path(f"{year}.json")
    if not path.exists():
        return []
    return json.load(open(path, "r", encoding="utf-8"))


def save_snapshot(snapshot, year: int):
    path = Path(f"{year}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)


# -----------------------------
# Main generator
# -----------------------------
def generate_snapshot(year: int):

    timestamp = f"{year}-12-31T23:59:59Z"

    with open(MASTER_PAGES_FILE, "r", encoding="utf-8") as f:
        master_pages = json.load(f)

    snapshot = load_snapshot(year)

    completed = {p["title"] for p in snapshot}
    failed_pages = []

    pages = list(master_pages.values())
    total = len(pages)

    print(f"\nStarting snapshot for {year}")
    print(f"Total pages: {total}\n")

    for i, page in enumerate(pages, start=1):

        title = page["title"]

        if title in completed:
            continue

        print(f"[{i}/{total}] Processing: {title}")

        try:
            revision = get_revision_before(title, timestamp)

            if not revision:
                print("  No revision found")
                continue

            revid = revision["revid"]

            wikitext = get_revision_wikitext(revid)
            links = extract_links(wikitext)
            categories = get_categories(title)

            snapshot.append({
                "page_id": page["page_id"],
                "title": title,
                "links": links,
                "categories": categories,
                "date": revision["timestamp"],
                "revision_id": revid
            })

            # SAVE AFTER EVERY PAGE (critical reliability improvement)
            save_snapshot(snapshot, year)

            print(f"  Saved ({len(snapshot)} pages)")

        except Exception as e:
            print(f"  Failed: {title}")
            print(e)
            failed_pages.append(title)

        # rate limiting with jitter
        time.sleep(MIN_REQUEST_DELAY + random.uniform(0.1, JITTER))

    # final save
    save_snapshot(snapshot, year)

    # save failures
    with open(f"{year}_failed.json", "w", encoding="utf-8") as f:
        json.dump(failed_pages, f, indent=2, ensure_ascii=False)

    print(f"\nFinished {year}")
    print(f"Saved pages: {len(snapshot)}")
    print(f"Failed pages: {len(failed_pages)}")


# -----------------------------
if __name__ == "__main__":
    year = int(input("Enter year (e.g. 2020): "))
    generate_snapshot(year)