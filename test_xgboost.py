# test_xgboost.py

import pandas as pd
from sqlalchemy import create_engine
from evaluation import RollingCV
from models.xgboost_model import XGBoostModel

def get_sample_data(product_id=1, store_id=1):
    engine = create_engine("postgresql://admin:secret@db:5432/supply_chain")
    query = f"""
    SELECT transaction_date, units_sold
    FROM sales_transactions
    WHERE product_id = {product_id} AND store_id = {store_id}
    ORDER BY transaction_date
    """
    return pd.read_sql(query, engine)

def main():
    df = get_sample_data(product_id=1, store_id=1)
    
    cv = RollingCV(min_train_size=730, test_size=30, step=90)
    model = XGBoostModel()
    
    results = cv.evaluate_model(model, df)
    
    # Show feature importance
    importance = model.get_feature_importance()
    print("\n🔍 Top 10 Most Important Features:")
    print(importance.head(10).to_string(index=False))
    
    results.to_csv('results_xgboost.csv', index=False)
    print("\n💾 Results saved!")

if __name__ == "__main__":
    main()
    # test_xgboost.py
# Smoke test for XGBoostModel with RollingCV evaluation

import pandas as pd
from sqlalchemy import create_engine
from evaluation import RollingCV
from models.xgboost_model import XGBoostModel

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

if __name__ == "__main__":
    df = get_sample_data(product_id=1, store_id=1)

    # Rolling CV setup
    cv = RollingCV(min_train_size=730, test_size=30, step=90)

    print("=" * 60)
    print("MODEL: XGBoost")
    print("=" * 60)

    xgb = XGBoostModel()
    results = cv.evaluate_model(xgb, df)

    print("\n📊 Average Metrics:")
    print(f"   MAE:  {results['mae'].mean():.2f}")
    print(f"   RMSE: {results['rmse'].mean():.2f}")
    print(f"   MAPE: {results['mape'].mean():.2f}%")

    # Save results
    results.to_csv('results_xgboost.csv', index=False)
    print("\n💾 Results saved to results_xgboost.csv")
