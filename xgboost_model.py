# /app/models/xgboost_model.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import features

class XGBoostModel:
    """
    XGBoost model for time series forecasting.
    Uses engineered features from features.py.
    """

    def __init__(self, params=None):
        self.params = params or {
            "objective": "reg:squarederror",
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }
        self.model = None
        self.feature_names = None

    def fit(self, train_df, date_col="transaction_date", target_col="units_sold"):
        """
        Train XGBoost on engineered features.
        Accepts date_col and target_col to match the evaluation framework.
        """
        df_features = features.build_features(train_df, date_col=date_col, target_col=target_col)
        self.feature_names = features.get_feature_columns(df_features, target_col=target_col, date_col=date_col)

        X = df_features[self.feature_names]
        y = df_features[target_col]

        self.model = XGBRegressor(**self.params)

        # simple validation split for early stopping if enough rows
        if len(X) > 20:
            split_idx = int(len(X) * 0.9)
            self.model.fit(
                X.iloc[:split_idx], y.iloc[:split_idx],
                eval_set=[(X.iloc[split_idx:], y.iloc[split_idx:])],
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.model.fit(X, y)

        return self

    def predict(self, test_df, date_col="transaction_date", target_col="units_sold"):
        """
        Generate predictions for test data. Accepts date_col and target_col.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before predicting!")

        df_features = features.build_features(test_df, date_col=date_col, target_col=target_col)
        X_test = df_features[self.feature_names]

        preds = self.model.predict(X_test)
        preds = np.maximum(preds, 0)  # clip negatives

        return pd.Series(preds, index=test_df[date_col].values, name="predicted")

    def get_feature_importance(self):
        if self.model is None:
            return None
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
