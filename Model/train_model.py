from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from utils import load_data, add_features, clip_growth, get_feature_columns, build_folds, baseline_metrics

def train_regression_model():
    df = load_data()
    df, df_target = add_features(df)
    df_target = clip_growth(df_target)

    target_col = "pagerank_growth_clipped"
    feature_cols = get_feature_columns(df_target)
    folds = build_folds(df_target)

    best_model = None
    best_mae = float("inf")
    best_params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1}

    for y, train_mask, test_mask in folds:
        train = df_target.loc[train_mask]
        test = df_target.loc[test_mask]

        X_train = train[feature_cols].values
        y_train = train[target_col].values
        X_test = test[feature_cols].values
        y_test = test[target_col].values

        model = XGBRegressor(
            random_state=42,
            objective="reg:squarederror",
            **best_params
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        if mae < best_mae:
            best_mae = mae
            best_model = model

    return best_model, feature_cols, df, df_target

if __name__ == "__main__":
    model, feature_cols, df, df_target = train_regression_model()
    print("Model trained successfully.")
