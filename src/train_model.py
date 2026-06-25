import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load combined dataset from GitHub
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

# Training data (up to 2024)
train = data[data["year"] <= 2024].dropna(subset=["pagerank_next_year"])
test  = data[data["year"] == 2025].dropna(subset=["pagerank_next_year"])

X_train = train[feature_cols]
y_train = train["pagerank_next_year"]

X_test = test[feature_cols]
y_test = test["pagerank_next_year"]

# Train model
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
