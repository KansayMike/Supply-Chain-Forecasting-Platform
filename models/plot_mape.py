import matplotlib.pyplot as plt

models = ['ARIMA', 'XGBoost']
mape = [69.66, 23.15]  # replace with Prophet once fixed

plt.bar(models, mape, color=['steelblue','orange'])
plt.ylabel('MAPE (%)')
plt.title('Forecasting Model Comparison')
plt.show()
