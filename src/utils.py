import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ALL_YEARS_URL = "https://raw.githubusercontent.com/Damone-Washington19/Winmaxxers/main/data-prep/Features/all_years_features.csv"

def load_data():
    df = pd.read_csv(ALL_YEARS_URL)
    df = df.sort_values(["page_id", "year"]).reset_index(drop=True)
    return df

def add_features(df):
    df["pagerank_rank"] = df.groupby("year")["pagerank"].rank(ascending=False)
    df["pagerank_rank_percentile"] = df.groupby("year")["pagerank_rank"].transform(
        lambda r: 1 - (r - 1) / (r.max() - 1 if r.max() > 1 else 1)
    )

    df["pagerank_next_year"] = df.groupby("page_id")["pagerank"].shift(-1)
    df["year_next"] = df.groupby("page_id")["year"].shift(-1)

    mask = (df["pagerank_next_year"].notna()) & (df["year_next"] == df["year"] + 1)
    df_target = df.loc[mask].copy()

    df_target["pagerank_growth"] = (
        df_target["pagerank_next_year"] - df_target["pagerank"]
    ) / df_target["pagerank"]

    df_target["pagerank_lag_1"] = df_target.groupby("page_id")["pagerank"].shift(1)
    df_target["pagerank_lag_2"] = df_target.groupby("page_id")["pagerank"].shift(2)

    df_target = df_target.dropna(subset=["pagerank_lag_1", "pagerank_lag_2"])
    return df, df_target

def clip_growth(df_target):
    df_target["pagerank_growth_clipped"] = df_target.groupby("year")["pagerank_growth"]\
        .transform(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)))
    return df_target

def get_feature_columns(df_target):
    exclude = [
        "pagerank_growth", "pagerank_growth_clipped",
        "pagerank_next_year", "year_next",
        "page_id", "title", "year",
        "pagerank_rank", "pagerank_rank_percentile"
    ]
    return [c for c in df_target.columns if c not in exclude and df_target[c].dtype != "object"]

def build_folds(df_target):
    years = sorted(df_target["year"].unique())
    folds = []
    for y in years[:-1]:
        train_mask = df_target["year"] <= y
        test_mask = df_target["year"] == (y + 1)
        if df_target[test_mask].empty:
            continue
        folds.append((y, train_mask, test_mask))
    return folds

def baseline_metrics(y_true):
    y_pred = np.zeros_like(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2
