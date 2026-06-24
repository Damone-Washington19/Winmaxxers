import pandas as pd

def load_data():
    url = "https://raw.githubusercontent.com/Damone-Washington19/Winmaxxers/main/data-prep/Features/years_features.csv"
    return pd.read_csv(url)

def get_feature_columns():
    return [
        "in_degree", "out_degree", "total_degree", "pagerank",
        "betweenness_centrality", "closeness_centrality",
        "eigenvector_centrality", "katz_centrality",
        "hub_score", "authority_score", "clustering_coefficient",
        "core_number", "avg_neighbor_degree", "avg_neighbor_pagerank",
        "community_size", "bridging_score"
    ]
