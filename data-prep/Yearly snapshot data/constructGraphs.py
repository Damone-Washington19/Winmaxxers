import json
from pathlib import Path

import networkx as nx
import pandas as pd
import community as community_louvain


YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

SCRIPT_DIR = Path(__file__).resolve().parent

MASTER_FILE = SCRIPT_DIR / "master_pages.json"


def load_master_titles():
    with open(MASTER_FILE, "r", encoding="utf-8") as f:
        master_data = json.load(f)

    return {
        page_info["title"]
        for page_info in master_data.values()
        if "title" in page_info
    }


def build_graph(snapshot_file):

    with open(snapshot_file, "r", encoding="utf-8") as f:
        pages = json.load(f)

    master_titles = load_master_titles()

    filtered_pages = [
        page
        for page in pages
        if page["title"] in master_titles
    ]

    filtered_titles = {
        page["title"]
        for page in filtered_pages
    }

    if len(filtered_pages) != len(pages):
        print(
            f"Filtered {len(pages) - len(filtered_pages)} pages not in master_pages.json"
        )

    G = nx.DiGraph()

    # Add nodes
    for page in filtered_pages:
        G.add_node(
            page["title"],
            page_id=page["page_id"]
        )

    edges = []

    # Add edges
    for page in filtered_pages:

        source = page["title"]

        for target in page.get("links", []):

            if target == source:
                continue

            if target not in filtered_titles:
                continue

            G.add_edge(source, target)
            edges.append((source, target))

    return G, filtered_pages, edges


def save_nodes(pages, year):

    rows = []

    for page in pages:
        rows.append({
            "page_id": page["page_id"],
            "title": page["title"]
        })

    pd.DataFrame(rows).to_csv(
        SCRIPT_DIR / f"{year}_nodes.csv",
        index=False
    )


def save_edges(edges, year):

    pd.DataFrame(
        edges,
        columns=["source", "target"]
    ).to_csv(
        SCRIPT_DIR / f"{year}_edges.csv",
        index=False
    )


def generate_features(G, pages, year):

    print("Computing PageRank...")

    pagerank = nx.pagerank(
        G,
        alpha=0.85
    )

    print("Computing communities...")

    UG = G.to_undirected()

    partition = community_louvain.best_partition(
        UG,
        random_state=42
    )

    community_sizes = {}

    for node, cid in partition.items():
        community_sizes[cid] = (
            community_sizes.get(cid, 0) + 1
        )

    rows = []

    for page in pages:

        title = page["title"]

        rows.append({

            "page_id":
                page["page_id"],

            "title":
                title,

            "in_degree":
                G.in_degree(title),

            "out_degree":
                G.out_degree(title),

            "total_degree":
                G.degree(title),

            "pagerank":
                pagerank.get(title, 0),

            "community_id":
                partition.get(title, -1),

            "community_size":
                community_sizes.get(
                    partition.get(title, -1),
                    0
                )
        })

    df = pd.DataFrame(rows)

    df.to_csv(
        SCRIPT_DIR / f"{year}_features.csv",
        index=False
    )

    print(
        f"Saved {year}_features.csv"
    )


def process_year(year):

    snapshot = SCRIPT_DIR / f"{year}.json"

    if not snapshot.exists():
        print(f"Missing {snapshot}")
        return

    print(f"\nProcessing {year}")

    G, pages, edges = build_graph(
        snapshot
    )

    print(
        f"Nodes: {G.number_of_nodes()}"
    )

    print(
        f"Edges: {G.number_of_edges()}"
    )

    save_nodes(
        pages,
        year
    )

    save_edges(
        edges,
        year
    )

    generate_features(
        G,
        pages,
        year
    )

    print(
        f"Finished {year}"
    )


if __name__ == "__main__":

    for year in YEARS:
        process_year(year)