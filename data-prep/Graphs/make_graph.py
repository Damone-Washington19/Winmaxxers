from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

MAX_VISUAL_NODES = 40
MAX_VISUAL_EDGES = 120


def load_graph(year=2020):
    nodes_file = SCRIPT_DIR / f"{year}_nodes.csv"
    edges_file = SCRIPT_DIR / f"{year}_edges.csv"

    if not nodes_file.exists():
        raise FileNotFoundError(
            f"Missing nodes file: {nodes_file}"
        )

    if not edges_file.exists():
        raise FileNotFoundError(
            f"Missing edges file: {edges_file}"
        )

    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)

    G = nx.DiGraph()

    for _, row in nodes_df.iterrows():
        title = str(row["title"])
        G.add_node(title, page_id=row.get("page_id"))

    for _, row in edges_df.iterrows():
        source = str(row["source"])
        target = str(row["target"])

        if source == target:
            continue

        if source in G and target in G:
            G.add_edge(source, target)

    return G


def build_visual_subgraph(G, max_nodes=MAX_VISUAL_NODES, max_edges=MAX_VISUAL_EDGES):
    if G.number_of_nodes() <= max_nodes and G.number_of_edges() <= max_edges:
        return G.copy()

    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=lambda n: degrees[n], reverse=True)[:max_nodes]
    subgraph = G.subgraph(top_nodes).copy()

    if subgraph.number_of_edges() <= max_edges:
        return subgraph

    edge_scores = []
    for u, v in subgraph.edges():
        score = degrees.get(u, 0) + degrees.get(v, 0)
        edge_scores.append((score, u, v))

    edge_scores.sort(reverse=True)
    top_edges = edge_scores[:max_edges]

    trimmed = nx.DiGraph()
    trimmed.add_nodes_from(subgraph.nodes(data=True))
    trimmed.add_edges_from((u, v) for _, u, v in top_edges)

    return trimmed


def draw_graph(G, year=2020, output_file="2020_graph.png"):
    if G.number_of_nodes() > MAX_VISUAL_NODES or G.number_of_edges() > MAX_VISUAL_EDGES:
        print(
            f"Graph too large ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)."
        )
        print("Using reduced subgraph for visualization.")
        G = build_visual_subgraph(G)

    plt.figure(figsize=(14, 10))

    pos = nx.spring_layout(G, seed=42, k=0.6)

    node_sizes = [300 + 150 * G.degree(node) for node in G.nodes()]
    edge_widths = [0.9 for _ in G.edges()]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color="skyblue",
        alpha=0.9,
        linewidths=0.5,
        edgecolors="black",
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="-|>",
        arrowsize=10,
        width=edge_widths,
        edge_color="gray",
        alpha=0.7,
    )

    label_count = min(25, G.number_of_nodes())
    top_labels = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:label_count]
    labels = {node: node for node, _ in top_labels}

    nx.draw_networkx_labels(
        G,
        pos,
        labels,
        font_size=9,
        font_color="black",
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"),
    )

    plt.title(
        f"2020 Graph Visualization ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)",
        fontsize=16,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=220)
    print(f"Saved graph image to {output_file}")
    plt.show()


def main():
    G = load_graph(year=2020)
    draw_graph(G, year=2020)


if __name__ == "__main__":
    main()
