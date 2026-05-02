# models/prophet_model.py
import pandas as pd
import numpy as np
import holidays
from prophet import Prophet

class ProphetModel:
    """
    Prophet model with holiday effects.
    Public methods:
      - fit(train_df, date_col='transaction_date', target_col='units_sold')
      - predict(test_df, date_col='transaction_date')
    """

    def __init__(self, country='EC', yearly_seasonality=True, weekly_seasonality=True):
        self.country = country
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.model = None

    def _prepare_data(self, df, date_col, target_col):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        prophet_df = pd.DataFrame({'ds': df[date_col], 'y': df[target_col]})

        # Fill missing dates (Prophet expects continuous dates)
        full_dates = pd.date_range(start=prophet_df['ds'].min(), end=prophet_df['ds'].max(), freq='D')
        prophet_df = prophet_df.set_index('ds').reindex(full_dates).rename_axis('ds').reset_index()
        prophet_df.columns = ['ds', 'y']
        prophet_df['y'] = prophet_df['y'].fillna(0)

        return prophet_df

    def _get_holidays(self, start_date, end_date):
        """
        Return a DataFrame in Prophet holiday format for the configured country.
        """
        years = range(start_date.year, end_date.year + 1)
        country_holidays = holidays.country_holidays(self.country, years=years)

        if not country_holidays:
            return pd.DataFrame(columns=['holiday', 'ds', 'lower_window', 'upper_window'])

        holiday_items = list(country_holidays.items())
        holiday_df = pd.DataFrame({
            'holiday': [name for date, name in holiday_items],
            'ds': pd.to_datetime([date for date, name in holiday_items]),
            'lower_window': 0,
            'upper_window': 1
        })
        return holiday_df

    def _make_prophet(self, holidays_df):
        """
        Try to create a Prophet instance using CmdStanPy backend if available.
        If that fails, try default constructor and raise a helpful error if initialization fails.
        """
        # Preferred: explicit CMDSTANPY backend for performance and stability
        try:
            return Prophet(
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=False,
                holidays=holidays_df if not holidays_df.empty else None,
                interval_width=0.95,
                changepoint_prior_scale=0.05,
                stan_backend="CMDSTANPY"
            )
        except Exception:
            # Try without explicit backend (older installs)
            try:
                return Prophet(
                    yearly_seasonality=self.yearly_seasonality,
                    weekly_seasonality=self.weekly_seasonality,
                    daily_seasonality=False,
                    holidays=holidays_df if not holidays_df.empty else None,
                    interval_width=0.95,
                    changepoint_prior_scale=0.05
                )
            except Exception as e:
                raise RuntimeError(
                    "Prophet failed to initialize. Install and configure a Stan backend (CmdStanPy recommended). "
                    "Inside the container run: pip install cmdstanpy; python -c \"import cmdstanpy; cmdstanpy.install_cmdstan()\". "
                    "If building CmdStan fails due to missing build tools, install system build tools (e.g., build-essential / make / gcc) first."
                ) from e

    def fit(self, train_df, date_col='transaction_date', target_col='units_sold'):
        """
        Train Prophet on the provided training DataFrame.
        Accepts date_col and target_col to match evaluation framework.
        """
        print("   Fitting Prophet model...")

        prophet_df = self._prepare_data(train_df, date_col, target_col)
        holiday_df = self._get_holidays(prophet_df['ds'].min(), prophet_df['ds'].max())

        print(f"   Found {len(holiday_df)} holidays in training period")

        self.model = self._make_prophet(holiday_df)
        # Fit the model (Prophet expects columns 'ds' and 'y')
        self.model.fit(prophet_df)

        print("   Prophet model fitted successfully")
        return self

    def predict(self, test_df, date_col='transaction_date'):
        """
        Generate predictions for test_df. Returns a pd.Series indexed by pd.DatetimeIndex.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before predicting!")

        future_dates = pd.DataFrame({'ds': pd.to_datetime(test_df[date_col].values)})
        forecast = self.model.predict(future_dates)

        preds = forecast['yhat'].values
        preds = np.maximum(preds, 0)

        index = pd.to_datetime(test_df[date_col]).reset_index(drop=True)
        return pd.Series(preds, index=index, name='predicted')

    def get_components(self):
        """
        Optionally return components (trend, seasonality, holidays) after fitting.
        """
        if self.model is None:
            return None
        # Prophet exposes components via predict on a history/future frame; return None here unless needed.
        return None
