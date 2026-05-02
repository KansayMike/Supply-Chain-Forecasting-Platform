# compare_models.py
# Head-to-head comparison: ARIMA vs Prophet

import pandas as pd
from sqlalchemy import create_engine
from evaluation import RollingCV
from models.arima_model import ARIMAModel
from models.prophet_model import ProphetModel

def get_sample_data(product_id=1, store_id=1):
    """Fetches one time series from the database."""
    engine = create_engine("postgresql://admin:secret@db:5432/supply_chain")
    
    query = f"""
    SELECT transaction_date, units_sold
    FROM sales_transactions
    WHERE product_id = {product_id} AND store_id = {store_id}
    ORDER BY transaction_date
    """
    
    return pd.read_sql(query, engine)

def run_comparison():
    df = get_sample_data(product_id=1, store_id=1)
    
    # Same cross-validation setup for both models (fair comparison!)
    cv = RollingCV(min_train_size=730, test_size=30, step=90)
    
    print("=" * 60)
    print("MODEL 1: ARIMA")
    print("=" * 60)
    arima = ARIMAModel(seasonal=False)
    arima_results = cv.evaluate_model(arima, df)
    
    print("\n" + "=" * 60)
    print("MODEL 2: Prophet (with holidays)")
    print("=" * 60)
    prophet = ProphetModel(country='EC')
    prophet_results = cv.evaluate_model(prophet, df)
    
    # Combine and compare
    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 60)
    
    comparison = pd.DataFrame({
        'ARIMA': [
            arima_results['mae'].mean(),
            arima_results['rmse'].mean(),
            arima_results['mape'].mean()
        ],
        'Prophet': [
            prophet_results['mae'].mean(),
            prophet_results['rmse'].mean(),
            prophet_results['mape'].mean()
        ]
    }, index=['mae', 'rmse', 'mape'])
    
    print(comparison.round(2))
    
    # Determine winner by lowest MAPE
    winner = comparison.loc['mape'].idxmin()
    print(f"\n🏆 Winner by MAPE: {winner}")
    improvement = abs(comparison.loc['mape']['ARIMA'] - comparison.loc['mape']['Prophet'])
    print(f"   Improvement: {improvement:.2f}%")
    
    # Save results
    arima_results.to_csv('results_arima.csv', index=False)
    prophet_results.to_csv('results_prophet.csv', index=False)
    comparison.to_csv('model_comparison.csv')
    
    print("\n💾 All results saved!")

if __name__ == "__main__":
    run_comparison()
