from pathlib import Path

import networkx as nx
import pandas as pd
import community as community_louvain


YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]


def build_graph(nodes_file, edges_file):

    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)

    G = nx.DiGraph()

    print("Adding nodes...")

    for _, row in nodes_df.iterrows():

        G.add_node(
            row["title"],
            page_id=row["page_id"]
        )

    print("Adding edges...")

    for _, row in edges_df.iterrows():

        source = row["source"]
        target = row["target"]

        if source not in G:
            continue

        if target not in G:
            continue

        G.add_edge(source, target)

    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        print(
            f"Removing {len(self_loops)} self-loop(s)"
        )
        G.remove_edges_from(self_loops)

    return G, nodes_df


def compute_features(G):

    print("Computing PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85)

    print("Computing Betweenness Centrality...")
    betweenness = nx.betweenness_centrality(
        G,
        normalized=True
    )

    print("Computing Closeness Centrality...")
    closeness = nx.closeness_centrality(G)

    print("Computing Eigenvector Centrality...")
    try:
        eigenvector = nx.eigenvector_centrality_numpy(
            G.to_undirected()
        )
    except Exception:
        eigenvector = {
            node: 0
            for node in G.nodes()
        }

    print("Computing Katz Centrality...")
    try:
        katz = nx.katz_centrality_numpy(
            G.to_undirected(),
            alpha=0.005
        )
    except Exception:
        katz = {
            node: 0
            for node in G.nodes()
        }

    print("Computing HITS Scores...")
    try:
        hubs, authorities = nx.hits(
            G,
            max_iter=1000,
            normalized=True
        )
    except Exception:
        hubs = {
            node: 0
            for node in G.nodes()
        }

        authorities = {
            node: 0
            for node in G.nodes()
        }

    UG = G.to_undirected()

    print("Computing Clustering Coefficient...")
    clustering = nx.clustering(UG)

    print("Computing Core Numbers...")
    core_numbers = nx.core_number(UG)

    print("Computing Average Neighbor Degree...")
    avg_neighbor_degree = nx.average_neighbor_degree(G)

    print("Computing Communities...")

    partition = community_louvain.best_partition(
        UG,
        random_state=42
    )

    community_sizes = {}

    for cid in partition.values():

        community_sizes[cid] = (
            community_sizes.get(cid, 0) + 1
        )

    print("Computing Average Neighbor PageRank...")

    avg_neighbor_pagerank = {}

    for node in G.nodes():

        neighbors = set(
            G.predecessors(node)
        ).union(
            set(G.successors(node))
        )

        if len(neighbors) == 0:

            avg_neighbor_pagerank[node] = 0

        else:

            avg_neighbor_pagerank[node] = (
                sum(
                    pagerank.get(n, 0)
                    for n in neighbors
                )
                /
                len(neighbors)
            )

    return {
        "pagerank": pagerank,
        "betweenness": betweenness,
        "closeness": closeness,
        "eigenvector": eigenvector,
        "katz": katz,
        "hubs": hubs,
        "authorities": authorities,
        "clustering": clustering,
        "core_numbers": core_numbers,
        "avg_neighbor_degree": avg_neighbor_degree,
        "avg_neighbor_pagerank": avg_neighbor_pagerank,
        "partition": partition,
        "community_sizes": community_sizes
    }


def generate_feature_dataframe(
    G,
    nodes_df,
    metrics
):

    rows = []

    for _, row in nodes_df.iterrows():

        title = row["title"]

        degree = G.degree(title)

        betweenness = metrics[
            "betweenness"
        ].get(title, 0)

        rows.append({

            "page_id":
                row["page_id"],

            "title":
                title,

            "in_degree":
                G.in_degree(title),

            "out_degree":
                G.out_degree(title),

            "total_degree":
                degree,

            "pagerank":
                metrics["pagerank"].get(
                    title,
                    0
                ),

            "betweenness_centrality":
                betweenness,

            "closeness_centrality":
                metrics["closeness"].get(
                    title,
                    0
                ),

            "eigenvector_centrality":
                metrics["eigenvector"].get(
                    title,
                    0
                ),

            "katz_centrality":
                metrics["katz"].get(
                    title,
                    0
                ),

            "hub_score":
                metrics["hubs"].get(
                    title,
                    0
                ),

            "authority_score":
                metrics["authorities"].get(
                    title,
                    0
                ),

            "clustering_coefficient":
                metrics["clustering"].get(
                    title,
                    0
                ),

            "core_number":
                metrics[
                    "core_numbers"
                ].get(
                    title,
                    0
                ),

            "avg_neighbor_degree":
                metrics[
                    "avg_neighbor_degree"
                ].get(
                    title,
                    0
                ),

            "avg_neighbor_pagerank":
                metrics[
                    "avg_neighbor_pagerank"
                ].get(
                    title,
                    0
                ),

            "community_size":
                metrics[
                    "community_sizes"
                ].get(
                    metrics[
                        "partition"
                    ].get(
                        title,
                        -1
                    ),
                    0
                ),

            "bridging_score":
                (
                    betweenness
                    /
                    (degree + 1)
                )
        })

    return pd.DataFrame(rows)


SCRIPT_DIR = Path(__file__).resolve().parent


def process_year(year):

    nodes_file = SCRIPT_DIR / f"{year}_nodes.csv"
    edges_file = SCRIPT_DIR / f"{year}_edges.csv"

    if not nodes_file.exists():

        print(f"Missing {nodes_file}")
        return

    if not edges_file.exists():

        print(f"Missing {edges_file}")
        return

    print(f"\nProcessing {year}")

    G, nodes_df = build_graph(
        nodes_file,
        edges_file
    )

    print(
        f"Nodes: {G.number_of_nodes()}"
    )

    print(
        f"Edges: {G.number_of_edges()}"
    )

    metrics = compute_features(G)

    features_df = (
        generate_feature_dataframe(
            G,
            nodes_df,
            metrics
        )
    )

    output_file = (
        f"{year}_features.csv"
    )

    features_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved {output_file}"
    )


if __name__ == "__main__":

    for year in YEARS:

        process_year(year)