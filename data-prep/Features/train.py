from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent


TARGET = "future_pagerank_growth"

FEATURES = [

    # Structural

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

    "bridging_score",

    # Temporal

    "pagerank_growth_1y",
    "pagerank_growth_2y",

    "degree_growth_1y",
    "degree_growth_2y",

    "authority_growth_1y",

    "betweenness_growth_1y",

    "pagerank_momentum",
    "degree_momentum",

    "pagerank_acceleration",
    "degree_acceleration"
]


def load_dataset():

    filename = SCRIPT_DIR / "temporal_features.csv"

    if not filename.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {filename}. "
            "Run temporal feature generation first."
        )

    df = pd.read_csv(filename)

    print("\nDataset Shape")
    print(df.shape)

    print("\nRows Per Year")
    print(
        df["year"]
        .value_counts()
        .sort_index()
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if df.empty:
        raise ValueError(
            f"Loaded dataset is empty: {filename}"
        )

    return df


def prepare_datasets(df):

    # ---------------------
    # Rows with labels
    # ---------------------

    labeled_df = df[
        df[TARGET].notna()
    ].copy()

    # ---------------------
    # Prediction rows
    # ---------------------

    prediction_df = df[
        df["year"] == 2026
    ].copy()

    # ---------------------
    # Train
    # ---------------------

    train_df = labeled_df[
        labeled_df["year"] <= 2024
    ].copy()

    # ---------------------
    # Validation
    # ---------------------

    valid_df = labeled_df[
        labeled_df["year"] == 2025
    ].copy()

    return (
        train_df,
        valid_df,
        prediction_df
    )


def clean_features(df):

    X = df[FEATURES].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    return X


def train_model(train_df):

    X_train = clean_features(
        train_df
    )

    y_train = train_df[
        TARGET
    ]

    model = XGBRegressor(

        n_estimators=500,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="reg:squarederror",

        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(
    model,
    valid_df
):

    if len(valid_df) == 0:

        print(
            "\nNo validation rows."
        )

        return

    X_valid = clean_features(
        valid_df
    )

    y_valid = valid_df[
        TARGET
    ]

    predictions = model.predict(
        X_valid
    )

    mae = mean_absolute_error(
        y_valid,
        predictions
    )

    r2 = r2_score(
        y_valid,
        predictions
    )

    print("\nValidation Results")
    print("------------------")

    print(
        f"MAE: {mae:.6f}"
    )

    print(
        f"R² : {r2:.4f}"
    )


def save_feature_importance(
    model
):

    importance_df = pd.DataFrame({

        "feature":
            FEATURES,

        "importance":
            model.feature_importances_

    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
    )

    importance_df.to_csv(
        "feature_importance.csv",
        index=False
    )

    print(
        "\nTop 20 Features"
    )

    print(
        importance_df.head(20)
    )


def predict_2026(
    model,
    prediction_df
):

    if len(prediction_df) == 0:

        print(
            "\nNo 2026 rows found."
        )

        return

    X_pred = clean_features(
        prediction_df
    )

    results = (
        prediction_df.copy()
    )

    results[
        "predicted_growth"
    ] = model.predict(
        X_pred
    )

    results = (
        results
        .sort_values(
            "predicted_growth",
            ascending=False
        )
    )

    results.to_csv(

        "predicted_emerging_topics.csv",

        index=False
    )

    print(
        "\nTop Predicted Emerging Technologies"
    )

    print(
        "-----------------------------------"
    )

    print(

        results[
            [
                "title",
                "predicted_growth"
            ]
        ]
        .head(25)

    )

    return results


def main():

    print(
        "Loading temporal_features.csv..."
    )

    df = load_dataset()

    (
        train_df,
        valid_df,
        prediction_df

    ) = prepare_datasets(df)

    print("\nDataset Split")
    print("-------------")

    print(
        f"Train Rows: {len(train_df)}"
    )

    print(
        f"Validation Rows: {len(valid_df)}"
    )

    print(
        f"Prediction Rows: {len(prediction_df)}"
    )

    if len(train_df) == 0:

        raise ValueError(
            "No training rows found."
        )

    model = train_model(
        train_df
    )

    evaluate_model(
        model,
        valid_df
    )

    save_feature_importance(
        model
    )

    predict_2026(
        model,
        prediction_df
    )

    print(
        "\nSaved Files:"
    )

    print(
        "feature_importance.csv"
    )

    print(
        "predicted_emerging_topics.csv"
    )


if __name__ == "__main__":
    main()