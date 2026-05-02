# evaluation.py
# This file contains the "ruler" we use to measure all our models.
# Every model (ARIMA, Prophet, XGBoost) will be tested with the SAME ruler.

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

def mean_absolute_percentage_error(y_true, y_pred):
    """
    MAPE = average of |(actual - predicted) / actual|
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Find where actual values are not zero
    mask = y_true != 0

    if not mask.any():
        return np.nan  # All zeros, can't calculate MAPE

    percentage_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return np.mean(percentage_errors) * 100  # Convert to percentage


def calculate_metrics(y_true, y_pred):
    """
    Returns a dictionary of metrics for a single forecast.
    Cleans NaNs before computing metrics.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Drop NaNs
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
    }
    return metrics


class RollingCV:
    """
    Implements rolling cross-validation for time series.
    """

    def __init__(self, min_train_size=365, test_size=30, step=30):
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step = step

    def split(self, df, date_col='transaction_date'):
        """
        Generates train/test splits for rolling CV.
        """
        df = df.sort_values(date_col)
        start_date = df[date_col].min()
        end_date = df[date_col].max()

        current_train_end = start_date + timedelta(days=self.min_train_size)
        window_num = 1

        while current_train_end + timedelta(days=self.test_size) <= end_date:
            train_mask = df[date_col] < current_train_end
            train_df = df[train_mask].copy()

            test_start = current_train_end
            test_end = current_train_end + timedelta(days=self.test_size)
            test_mask = (df[date_col] >= test_start) & (df[date_col] < test_end)
            test_df = df[test_mask].copy()

            if len(train_df) > 0 and len(test_df) > 0:
                split_info = {
                    'window': window_num,
                    'train_start': train_df[date_col].min(),
                    'train_end': train_df[date_col].max(),
                    'test_start': test_df[date_col].min(),
                    'test_end': test_df[date_col].max(),
                    'train_rows': len(train_df),
                    'test_rows': len(test_df)
                }
                yield train_df, test_df, split_info
                window_num += 1

            current_train_end += timedelta(days=self.step)

    def evaluate_model(self, model, df, date_col='transaction_date', target_col='units_sold'):
        """
        Runs a model through all CV windows and collects metrics.
        """
        results = []
        print(f"🔬 Running rolling cross-validation...")

        for train_df, test_df, split_info in self.split(df, date_col):
            print(f"   Window {split_info['window']}: "
                  f"Train {split_info['train_start'].strftime('%Y-%m-%d')} to {split_info['train_end'].strftime('%Y-%m-%d')} | "
                  f"Test {split_info['test_start'].strftime('%Y-%m-%d')} to {split_info['test_end'].strftime('%Y-%m-%d')}")

            # Fit model
            model.fit(train_df, date_col=date_col, target_col=target_col)

            # Predict
            predictions = model.predict(test_df, date_col=date_col)

            # Metrics
            y_true = test_df[target_col].values
            y_pred = predictions.values
            metrics = calculate_metrics(y_true, y_pred)
            metrics.update(split_info)

            results.append(metrics)

        results_df = pd.DataFrame(results)

        print(f"✅ Completed {len(results_df)} windows")
        print(f"\n📊 Average Metrics:")
        print(f"   MAE:  {results_df['mae'].mean():.2f}")
        print(f"   RMSE: {results_df['rmse'].mean():.2f}")
        print(f"   MAPE: {results_df['mape'].mean():.2f}%")

        return results_df
