# models/arima_model.py
# Auto-ARIMA forecasting model

import pandas as pd
from pmdarima import auto_arima


class ARIMAModel:
    """
    ARIMA forecasting model with automatic parameter selection.
    Uses pmdarima.auto_arima to search for the best (p,d,q) order.
    """

    def __init__(self, seasonal: bool = False, seasonal_period: int = 7):
        """
        Parameters
        ----------
        seasonal : bool
            Whether to include seasonal component (SARIMA).
        seasonal_period : int
            Seasonal cycle length (e.g., 7 = weekly, 365 = yearly).
        """
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.fitted_model = None

    def fit(
        self,
        train_df: pd.DataFrame,
        date_col: str = "transaction_date",
        target_col: str = "units_sold",
    ):
        """
        Fit the ARIMA model on historical data.

        Steps:
        1. Sort by date and set index.
        2. Extract target series and ensure daily frequency.
        3. Run auto_arima to find best parameters.
        """
        print("   Fitting ARIMA model...")

        df = train_df.copy().sort_values(date_col).set_index(date_col)
        y = df[target_col].asfreq("D", fill_value=0)

        print("   Searching for optimal parameters...")
        try:
            self.fitted_model = auto_arima(
                y,
                seasonal=self.seasonal,
                m=self.seasonal_period if self.seasonal else 1,
                start_p=0,
                max_p=5,
                start_q=0,
                max_q=5,
                max_d=3,  # allow more differencing
                d=None,
                trace=False,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )
            print(f"   Best model: ARIMA{self.fitted_model.order}")
            if self.seasonal:
                print(f"   Seasonal: {self.fitted_model.seasonal_order}")
        except Exception as e:
            print("   ARIMA fitting failed:", e)
            self.fitted_model = None

        return self

    def predict(self, test_df: pd.DataFrame, date_col: str = "transaction_date") -> pd.Series:
        """
        Generate predictions for the test period.
        Returns a Series of predicted values aligned with test dates.
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before predicting!")

        n_periods = len(test_df)
        forecast = self.fitted_model.predict(n_periods=n_periods)

        predictions = pd.Series(forecast, index=test_df[date_col].values, name="predicted")
        return predictions.clip(lower=0)

    def get_params(self) -> dict | None:
        """Return fitted model parameters for documentation."""
        if self.fitted_model is None:
            return None
        return {
            "order": self.fitted_model.order,
            "seasonal_order": self.fitted_model.seasonal_order if self.seasonal else None,
            "aic": self.fitted_model.aic(),
        }
