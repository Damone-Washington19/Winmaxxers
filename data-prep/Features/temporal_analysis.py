from pathlib import Path

import numpy as np
import pandas as pd


YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

EPS = 1e-9


FEATURE_FILES = [
    f"{year}_features.csv"
    for year in YEARS
]


BASE_FEATURES = [

    "in_degree",
    "out_degree",
    "total_degree",

    "pagerank",

    "betweenness_centrality",
    "closeness_centrality",
    "eigenvector_centrality",
    "katz_centrality",

    "hub_score",
    "authority_score",

    "clustering_coefficient",

    "core_number",

    "avg_neighbor_degree",
    "avg_neighbor_pagerank",

    "community_size",

    "bridging_score"
]

SCRIPT_DIR = Path(__file__).resolve().parent


def load_feature_files():

    dfs = []

    for year in YEARS:

        filename = SCRIPT_DIR / f"{year}_features.csv"

        if not filename.exists():

            print(f"Missing {filename}")
            continue

        df = pd.read_csv(filename)
        df["year"] = year
        dfs.append(df)

    if not dfs:
        raise ValueError(
            "No feature files found. "
            "Please run feature extraction first."
        )

    return pd.concat(
        dfs,
        ignore_index=True
    )


def safe_growth(current, previous):

    return (
        current - previous
    ) / (
        abs(previous) + EPS
    )


def build_temporal_features(df):

    df = df.sort_values(
        ["title", "year"]
    )

    grouped = []

    for title, group in df.groupby(
        "title",
        sort=False
    ):

        group = group.sort_values(
            "year"
        ).copy()

        # -------------------------
        # Previous Year Features
        # -------------------------

        group[
            "prev_pagerank"
        ] = group[
            "pagerank"
        ].shift(1)

        group[
            "prev_degree"
        ] = group[
            "total_degree"
        ].shift(1)

        group[
            "prev_authority"
        ] = group[
            "authority_score"
        ].shift(1)

        group[
            "prev_betweenness"
        ] = group[
            "betweenness_centrality"
        ].shift(1)

        # -------------------------
        # Growth Features
        # -------------------------

        group[
            "pagerank_growth_1y"
        ] = safe_growth(
            group["pagerank"],
            group["prev_pagerank"]
        )

        group[
            "degree_growth_1y"
        ] = safe_growth(
            group["total_degree"],
            group["prev_degree"]
        )

        group[
            "authority_growth_1y"
        ] = safe_growth(
            group["authority_score"],
            group["prev_authority"]
        )

        group[
            "betweenness_growth_1y"
        ] = safe_growth(
            group[
                "betweenness_centrality"
            ],
            group[
                "prev_betweenness"
            ]
        )

        # -------------------------
        # Two-Year Growth
        # -------------------------

        pr_2y_prev = (
            group["pagerank"]
            .shift(2)
        )

        deg_2y_prev = (
            group["total_degree"]
            .shift(2)
        )

        group[
            "pagerank_growth_2y"
        ] = safe_growth(
            group["pagerank"],
            pr_2y_prev
        )

        group[
            "degree_growth_2y"
        ] = safe_growth(
            group["total_degree"],
            deg_2y_prev
        )

        # -------------------------
        # Momentum
        # -------------------------

        group[
            "pagerank_momentum"
        ] = (
            group["pagerank"]
            -
            pr_2y_prev
        )

        group[
            "degree_momentum"
        ] = (
            group["total_degree"]
            -
            deg_2y_prev
        )

        # -------------------------
        # Acceleration
        # -------------------------

        group[
            "pagerank_acceleration"
        ] = (
            group[
                "pagerank_growth_1y"
            ]
            -
            group[
                "pagerank_growth_1y"
            ].shift(1)
        )

        group[
            "degree_acceleration"
        ] = (
            group[
                "degree_growth_1y"
            ]
            -
            group[
                "degree_growth_1y"
            ].shift(1)
        )

        # -------------------------
        # Future Label
        # -------------------------

        future_pr = (
            group["pagerank"]
            .shift(-1)
        )

        group[
            "future_pagerank_growth"
        ] = safe_growth(
            future_pr,
            group["pagerank"]
        )

        grouped.append(group)

    final_df = pd.concat(
        grouped,
        ignore_index=True
    )

    # remove rows without
    # enough history

    final_df = final_df.dropna(
        subset=[

            "pagerank_growth_1y",

            "degree_growth_1y",

            "authority_growth_1y",

            "betweenness_growth_1y",

            "future_pagerank_growth"
        ]
    )

    return final_df


def main():

    print(
        "Loading yearly feature files..."
    )

    df = load_feature_files()

    print(
        f"Loaded {len(df)} rows"
    )

    print(
        "Generating temporal features..."
    )

    temporal_df = (
        build_temporal_features(df)
    )

    temporal_df.to_csv(
        "temporal_features.csv",
        index=False
    )

    print(
        "\nSaved temporal_features.csv"
    )

    print(
        f"Rows: {len(temporal_df)}"
    )

    print(
        f"Columns: {len(temporal_df.columns)}"
    )


if __name__ == "__main__":
    main()
