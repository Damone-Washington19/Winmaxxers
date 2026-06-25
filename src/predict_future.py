import pandas as pd
from utils import load_data, add_features, get_feature_columns
from train_model import train_regression_model

def predict_next_year():
    model, feature_cols, df, df_target = train_regression_model()

    latest_year = df_target["year"].max()
    df_current = df[df["year"] == latest_year].copy()

    df_sorted = df.sort_values(["page_id", "year"]).copy()
    df_sorted["pagerank_lag_1"] = df_sorted.groupby("page_id")["pagerank"].shift(1)
    df_sorted["pagerank_lag_2"] = df_sorted.groupby("page_id")["pagerank"].shift(2)

    df_current = df_current.merge(
        df_sorted[["page_id", "year", "pagerank_lag_1", "pagerank_lag_2"]],
        on=["page_id", "year"],
        how="left"
    ).dropna(subset=["pagerank_lag_1", "pagerank_lag_2"])

    X_curr = df_current[feature_cols].values
    df_current["predicted_growth"] = model.predict(X_curr)
    df_current["predicted_pagerank_next"] = df_current["pagerank"] * (1 + df_current["predicted_growth"])

    return df_current

if __name__ == "__main__":
    preds = predict_next_year()
    print(preds.head())
