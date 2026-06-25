import numpy as np
from train_model import train_regression_model
from utils import load_data, add_features, get_feature_columns

def recursive_forecast(start_year, end_year=2031):
    model, feature_cols, df, df_target = train_regression_model()

    df_current = df[df["year"] == start_year].copy()

    df_sorted = df.sort_values(["page_id", "year"]).copy()
    df_sorted["pagerank_lag_1"] = df_sorted.groupby("page_id")["pagerank"].shift(1)
    df_sorted["pagerank_lag_2"] = df_sorted.groupby("page_id")["pagerank"].shift(2)

    df_current = df_current.merge(
        df_sorted[["page_id", "year", "pagerank_lag_1", "pagerank_lag_2"]],
        on=["page_id", "year"],
        how="left"
    ).dropna(subset=["pagerank_lag_1", "pagerank_lag_2"])

    structural_features = [
        c for c in feature_cols
        if "pagerank" not in c and "lag" not in c
    ]

    df_hist = df.sort_values(["page_id", "year"]).copy()
    yoY_change = {}
    for feat in structural_features:
        df_hist[f"{feat}_pct"] = df_hist.groupby("page_id")[feat].pct_change()
        yoY_change[feat] = df_hist[f"{feat}_pct"].mean()

    historical_rmse = 0.1  # heuristic

    forecasts = []
    curr = df_current.copy()

    for step, year in enumerate(range(start_year + 1, end_year + 1), start=1):
        X = curr[feature_cols].values
        curr["predicted_growth"] = model.predict(X)
        curr["pagerank"] *= (1 + curr["predicted_growth"])
        curr["year"] = year

        for feat in structural_features:
            curr[feat] *= (1 + yoY_change.get(feat, 0))

        uncertainty = historical_rmse * step

        for _, row in curr.iterrows():
            forecasts.append({
                "page_id": row["page_id"],
                "title": row["title"],
                "year": year,
                "predicted_pagerank": row["pagerank"],
                "predicted_growth": row["predicted_growth"],
                "uncertainty": uncertainty
            })

    return forecasts

if __name__ == "__main__":
    out = recursive_forecast(2026)
    print(out[:5])
