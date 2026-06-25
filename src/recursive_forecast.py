import pandas as pd
from xgboost import XGBRegressor

# Load data from GitHub
DATA_URL = "https://raw.githubusercontent.com/Damone-Washington19/Winmaxxers/main/data-prep/Features/years_features.csv"
data = pd.read_csv(DATA_URL)

# Sort and create next-year label
data = data.sort_values(["page_id", "year"])
data["pagerank_next_year"] = data.groupby("page_id")["pagerank"].shift(-1)

# Feature list
FEATURE_COLS = [
    "in_degree", "out_degree", "total_degree", "pagerank",
    "betweenness_centrality", "closeness_centrality",
    "eigenvector_centrality", "katz_centrality",
    "hub_score", "authority_score", "clustering_coefficient",
    "core_number", "avg_neighbor_degree", "avg_neighbor_pagerank",
    "community_size", "bridging_score"
]

def train_base_model():
    """Train the one-year-ahead model on all available historical data."""
    train = data.dropna(subset=["pagerank_next_year"])
    X_train = train[FEATURE_COLS]
    y_train = train["pagerank_next_year"]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model

def recursive_forecast(start_year=2026, steps=5):
    """
    Predict multiple years into the future using recursive forecasting.
    start_year = last real year in the dataset
    steps = how many years ahead to predict
    """
    model = train_base_model()

    # Start with the last real year
    current = data[data["year"] == start_year].copy()
    forecasts = []

    for i in range(1, steps + 1):
        future_year = start_year + i

        # Predict next-year PageRank
        current["predicted_pagerank"] = model.predict(current[FEATURE_COLS])

        # Save results
        result = current[["page_id", "title"]].copy()
        result["year"] = future_year
        result["predicted_pagerank"] = current["predicted_pagerank"]
        forecasts.append(result)

        # Prepare for next iteration
        current = current.copy()
        current["pagerank"] = current["predicted_pagerank"]

    return pd.concat(forecasts, ignore_index=True)

if __name__ == "__main__":
    forecast = recursive_forecast(start_year=2026, steps=5)
    print(forecast.sort_values(["year", "predicted_pagerank"], ascending=[True, False]).head(50))
