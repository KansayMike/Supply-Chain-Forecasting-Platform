# features.py
# Feature engineering pipeline for time series forecasting

import pandas as pd
import numpy as np
import holidays

def create_features(df, date_col='transaction_date', target_col='units_sold'):
    """
    Takes raw sales data and creates all features XGBoost needs.
    """
    df = df.copy()
    df = df.sort_values(date_col)
    df[date_col] = pd.to_datetime(df[date_col])

    # --- Lag features ---
    for lag in [1, 7, 14, 21, 28]:
        df[f'sales_lag_{lag}'] = df[target_col].shift(lag)

    # --- Rolling statistics ---
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}d'] = df[target_col].shift(1).rolling(window, min_periods=1).mean()
        df[f'rolling_std_{window}d']  = df[target_col].shift(1).rolling(window, min_periods=1).std()
        df[f'rolling_max_{window}d']  = df[target_col].shift(1).rolling(window, min_periods=1).max()
        df[f'rolling_sum_{window}d']  = df[target_col].shift(1).rolling(window, min_periods=1).sum()

    # --- Date/time features ---
    df['day_of_week']   = df[date_col].dt.dayofweek
    df['day_of_month']  = df[date_col].dt.day
    df['month']         = df[date_col].dt.month
    df['quarter']       = df[date_col].dt.quarter
    df['year']          = df[date_col].dt.year
    df['is_weekend']    = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start']= (df['day_of_month'] <= 5).astype(int)
    df['is_month_end']  = (df['day_of_month'] >= 25).astype(int)

    # --- Difference features ---
    df['sales_diff_7d']  = df[target_col] - df['sales_lag_7']
    df['sales_diff_14d'] = df[target_col] - df['sales_lag_14']
    df['sales_pct_change_7d'] = (df[target_col] - df['sales_lag_7']) / (df['sales_lag_7'] + 1)

    # --- External regressors (holidays) ---
    # Use Ecuador holidays as example; convert holiday dates to pandas Timestamps
    ec_holidays = holidays.country_holidays('EC', years=range(2013, 2018))
    ec_holidays_dates = sorted(pd.to_datetime(list(ec_holidays.keys())))

    # is_holiday: compare normalized timestamps
    df['is_holiday'] = df[date_col].dt.normalize().isin(ec_holidays_dates).astype(int)

    # helper functions that work with pd.Timestamp
    def _days_to_holiday(ts):
        ts = pd.Timestamp(ts).normalize()
        future = [d for d in ec_holidays_dates if d >= ts]
        return int((future[0] - ts).days) if future else 365

    def _days_since_holiday(ts):
        ts = pd.Timestamp(ts).normalize()
        past = [d for d in ec_holidays_dates if d <= ts]
        return int((ts - past[-1]).days) if past else 365

    df['days_to_holiday'] = df[date_col].apply(_days_to_holiday)
    df['days_since_holiday'] = df[date_col].apply(_days_since_holiday)

    # --- Expanding statistics ---
    df['expanding_mean'] = df[target_col].shift(1).expanding().mean()
    df['expanding_std']  = df[target_col].shift(1).expanding().std()

    # --- Cleanup ---
    df = df.fillna(0)
    return df


def get_feature_columns(df, target_col='units_sold', date_col='transaction_date'):
    """
    Returns list of feature column names (everything except target and date).
    """
    exclude = [target_col, date_col, 'product_id', 'store_id']
    return [col for col in df.columns if col not in exclude]


def build_features(df, date_col='transaction_date', target_col='units_sold'):
    """
    Wrapper to call create_features and return engineered DataFrame.
    """
    return create_features(df, date_col=date_col, target_col=target_col)
