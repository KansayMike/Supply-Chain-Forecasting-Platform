"""test_arima.py
Smoke test that prints progress messages similar to the example output.
Assumes a CSV named data.csv in the same folder with either:
- columns: date,value  (date parseable by pandas)
or
- a single column of numeric values (no header or header is value)
"""
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import csv
import logging

from models import ARIMAModel
from sklearn.metrics import mean_absolute_error, mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def load_series(path):
    df = pd.read_csv(path)
    # try date,value
    if "date" in df.columns and "value" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        series = df["value"]
        dates = df["date"]
    elif df.shape[1] == 1:
        series = df.iloc[:,0].dropna().reset_index(drop=True)
        # fabricate daily dates for display
        start = datetime(2013,1,1)
        dates = pd.date_range(start, periods=len(series), freq="D")
    else:
        # fallback: try common names
        for c in ("value","y","demand","sales"):
            if c in df.columns:
                series = df[c].dropna().reset_index(drop=True)
                start = pd.to_datetime(df.columns[0]) if "date" in df.columns else datetime(2013,1,1)
                dates = pd.date_range(start, periods=len(series), freq="D")
                break
        else:
            raise ValueError("CSV format not recognized. Provide date,value or single value column.")
    return pd.Series(series).reset_index(drop=True), pd.Series(dates)

def rolling_cv_print(series, dates, initial_window=365, horizon=30, max_windows=5):
    n = len(series)
    windows = 0
    results = []
    print(f"📊 Loaded {n} days of data for product 1, store 1")
    print(f"   Date range: {dates.iloc[0].date()} to {dates.iloc[-1].date()}")
    print("🔬 Running rolling cross-validation...")
    fold = 0
    while (initial_window + fold*horizon + horizon) <= n and (max_windows is None or windows < max_windows):
        train_end = initial_window + fold*horizon
        test_start = train_end
        test_end = train_end + horizon
        train = series.iloc[:train_end]
        test = series.iloc[test_start:test_end]
        train_start_date = dates.iloc[0].date()
        train_end_date = dates.iloc[train_end-1].date()
        test_start_date = dates.iloc[test_start].date()
        test_end_date = dates.iloc[test_end-1].date()
        windows += 1
        print(f"   Window {windows}: Train {train_start_date} to {train_end_date} | Test {test_start_date} to {test_end_date}")
        print("   Fitting ARIMA model...")
        print("   Searching for optimal parameters...")
        try:
            wrapper = ARIMAModel(seasonal=False, m=1, arima_kwargs={'max_p':3,'max_q':3,'max_d':2})
            wrapper.fit(train)
            # best model info if available
            order = getattr(wrapper.model, 'order', None)
            if order:
                print(f"   Best model: ARIMA{order}")
            preds = wrapper.predict(len(test))
        except Exception as e:
            print("   ARIMA failed on this window; using naive forecast.")
            preds = np.repeat(train.iloc[-1], len(test))
            order = None

        mae = mean_absolute_error(test, preds)
        r_mse = rmse(test, preds)
        mape = np.mean(np.abs((test - preds) / np.where(test==0, 1e-8, test))) * 100.0

        results.append({
            "window": windows,
            "train_start": str(train_start_date),
            "train_end": str(train_end_date),
            "test_start": str(test_start_date),
            "test_end": str(test_end_date),
            "order": str(order),
            "mae": float(mae),
            "rmse": float(r_mse),
            "mape": float(mape)
        })
        fold += 1

    print(f"✅ Completed {len(results)} windows")
    # aggregate
    maes = [r["mae"] for r in results]
    rmses = [r["rmse"] for r in results]
    mapes = [r["mape"] for r in results]
    if maes:
        print("📊 Average Metrics:")
        print(f"   MAE:  {np.mean(maes):.2f}")
        print(f"   RMSE: {np.mean(rmses):.2f}")
        print(f"   MAPE: {np.mean(mapes):.2f}%")
    else:
        print("No windows were run; check series length and initial_window/horizon.")

    # save results
    out = Path("arima_results.csv")
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["window","train_start","train_end","test_start","test_end","order","mae","rmse","mape"])
        for r in results:
            writer.writerow([r["window"], r["train_start"], r["train_end"], r["test_start"], r["test_end"], r["order"], r["mae"], r["rmse"], r["mape"]])
    print(f"💾 Results saved to {out.name}")

def main():
    data_path = Path("data.csv")
    if not data_path.exists():
        # create a small synthetic dataset if none exists
        print("data.csv not found — creating a synthetic dataset for demo.")
        dates = pd.date_range("2013-01-01", periods=1684, freq="D")
        values = (np.sin(np.arange(len(dates))/200.0) * 100 + 300 + np.random.normal(0,20,len(dates))).round().astype(int)
        pd.DataFrame({"date": dates, "value": values}).to_csv("data.csv", index=False)
    series, dates = load_series("data.csv")
    # run rolling CV with smaller windows for demo
    rolling_cv_print(series, dates, initial_window=365*2, horizon=30, max_windows=5)

if __name__ == "__main__":
    main()
