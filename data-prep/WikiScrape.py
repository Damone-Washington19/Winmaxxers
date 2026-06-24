import json
import requests
import time

API_URL = "https://en.wikipedia.org/w/api.php"

ROOT_CATEGORY = "Category:Nanotechnology"

HEADERS = {
    "User-Agent": "WinmaxxersBot/1.0 (https://example.com; contact@example.com)",
    "Accept": "application/json"
}
MAX_RETRIES = 8
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
INITIAL_RETRY_DELAY = 2.0
REQUEST_DELAY = 1.0


def request_with_retries(params):
    delay = INITIAL_RETRY_DELAY

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

        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            delay = float(retry_after)

        print(
            f"Request {params.get('cmtitle', params.get('titles'))}: "
            f"status {response.status_code}, retrying in {delay:.1f}s..."
        )
        time.sleep(delay)
        delay = max(delay * 2, INITIAL_RETRY_DELAY)

    raise RuntimeError("Exceeded retry limit for Wikipedia API request")


def get_category_members(category):
    members = []
    cmcontinue = None

    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": "max"
        }

        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        response = request_with_retries(params)
        data = response.json()

        members.extend(data["query"]["categorymembers"])

        if "continue" not in data:
            break

        cmcontinue = data["continue"]["cmcontinue"]

        time.sleep(REQUEST_DELAY)

    return members


def is_valid_page(title):
    invalid_prefixes = (
        "Category:",
        "Template:",
        "Portal:",
        "File:",
        "Wikipedia:",
        "Help:"
    )

    if title.startswith(invalid_prefixes):
        return False

    if title.startswith("List of"):
        return False

    return True


def collect_categories(root_category, max_depth=2):
    """
    Breadth-first traversal of category tree.

    depth 0 = root category
    depth 1 = child categories
    depth 2 = grandchild categories
    """

    visited = set()
    categories = []

    queue = [(root_category, 0)]

    while queue:
        category, depth = queue.pop(0)

        if category in visited:
            continue

        visited.add(category)

        categories.append((category, depth))

        print(f"[Depth {depth}] {category}")

        if depth >= max_depth:
            continue

        members = get_category_members(category)

        for member in members:
            title = member["title"]

            if title.startswith("Category:"):
                queue.append((title, depth + 1))

    return categories


def build_master_list():
    pages = {}

    categories = collect_categories(
        ROOT_CATEGORY,
        max_depth=2
    )

    print(f"\nFound {len(categories)} categories\n")

    for category, depth in categories:

        members = get_category_members(category)

        for member in members:

            title = member["title"]

            if not is_valid_page(title):
                continue

            page_id = str(member["pageid"])

            if page_id not in pages:
                pages[page_id] = {
                    "page_id": member["pageid"],
                    "title": title,
                    "discovered_from": [category],
                    "category_depth": depth
                }

            else:
                pages[page_id]["discovered_from"].append(category)

    with open(
        "master_pages.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            pages,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nUnique pages saved: {len(pages)}")


if __name__ == "__main__":
    build_master_list()