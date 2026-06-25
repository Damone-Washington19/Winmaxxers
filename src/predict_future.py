import pandas as pd
from xgboost import XGBRegressor

# Load data
url = "https://raw.githubusercontent.com/Damone-Washington19/Winmaxxers/main/data-prep/Features/years_features.csv"
data = pd.read_csv(url)

# Sort and create next-year label
data = data.sort_values(["page_id", "year"])
data["pagerank_next_year"] = data.groupby("page_id")["pagerank"].shift(-1)

# Feature list
feature_cols = [
    "in_degree", "out_degree", "total_degree", "pagerank",
    "betweenness_centrality", "closeness_centrality",
    "eigenvector_centrality", "katz_centrality",
    "hub_score", "authority_score", "clustering_coefficient",
    "core_number", "avg_neighbor_degree", "avg_neighbor_pagerank",
    "community_size", "bridging_score"
]

# Train on all years except 2026
train = data[data["year"] <= 2025].dropna(subset=["pagerank_next_year"])
X_train = train[feature_cols]
y_train = train["pagerank_next_year"]

# Predict 2027 from 2026 features
future = data[data["year"] == 2026]
X_future = future[feature_cols]

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

future["predicted_2027_pagerank"] = model.predict(X_future)

# Sort by predicted importance
future = future.sort_values("predicted_2027_pagerank", ascending=False)

print(future[["page_id", "title", "predicted_2027_pagerank"]].head(20))
