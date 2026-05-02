import pandas as pd
from models import ARIMAModel

df = pd.read_csv("test_series.csv")
y = df["value"]

m = ARIMAModel(seasonal=False, m=1, arima_kwargs={"max_p":3, "max_q":3})
m.fit(y)
pred = m.predict(3)
print("Forecast 3 steps:", pred)
m.save("models/arima_baseline.joblib")
print("Saved model to models/arima_baseline.joblib")
