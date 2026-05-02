# final_comparison.py
# The grand finale: ARIMA vs Prophet vs XGBoost

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from evaluation import RollingCV
from models.arima_model import ARIMAModel
from models.prophet_model import ProphetModel
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

def run_all_models():
    df = get_sample_data(product_id=1, store_id=1)
    
    # Same CV for ALL models (fair fight!)
    cv = RollingCV(min_train_size=730, test_size=30, step=90)
    
    results = {}
    
    # ============================================================
    # MODEL 1: ARIMA
    # ============================================================
    print("\n" + "=" * 70)
    print("MODEL 1: ARIMA (Statistical Baseline)")
    print("=" * 70)
    arima = ARIMAModel(seasonal=False)
    results['ARIMA'] = cv.evaluate_model(arima, df)
    
    # ============================================================
    # MODEL 2: Prophet
    # ============================================================
    print("\n" + "=" * 70)
    print("MODEL 2: Prophet (Trend + Seasonality + Holidays)")
    print("=" * 70)
    prophet = ProphetModel(country='EC')
    results['Prophet'] = cv.evaluate_model(prophet, df)
    
    # ============================================================
    # MODEL 3: XGBoost
    # ============================================================
    print("\n" + "=" * 70)
    print("MODEL 3: XGBoost (Gradient Boosting with Features)")
    print("=" * 70)
    xgb_model = XGBoostModel()
    results['XGBoost'] = cv.evaluate_model(xgb_model, df)
    
    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    
    comparison = pd.DataFrame({
        model: {
            'MAE': res['mae'].mean(),
            'RMSE': res['rmse'].mean(),
            'MAPE (%)': res['mape'].mean(),
            'Std MAPE': res['mape'].std()  # Consistency measure
        }
        for model, res in results.items()
    })
    
    print(comparison.round(2))
    
    # ============================================================
    # STATISTICAL SIGNIFICANCE (Paired t-test)
    # ============================================================
    # We use paired t-test because the SAME test windows are used for all models
    # This tells us if the difference is real or just random luck
    
    from scipy import stats
    
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE (Paired t-test on MAPE)")
    print("=" * 70)
    print("p < 0.05 means the difference is statistically significant")
    print("-" * 70)
    
    models = list(results.keys())
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            # Paired t-test: same windows, different models
            t_stat, p_value = stats.ttest_rel(
                results[m1]['mape'],
                results[m2]['mape']
            )
            significance = "✅ SIGNIFICANT" if p_value < 0.05 else "❌ Not significant"
            print(f"{m1} vs {m2}: p = {p_value:.4f} {significance}")
    
    # ============================================================
    # WINNER SELECTION
    # ============================================================
    print("\n" + "=" * 70)
    print("WINNER SELECTION")
    print("=" * 70)
    
    # Rank by MAPE (lower is better)
    mape_scores = {model: res['mape'].mean() for model, res in results.items()}
    winner = min(mape_scores, key=mape_scores.get)
    
    print(f"🏆 Winner: {winner}")
    print(f"   MAPE: {mape_scores[winner]:.2f}%")
    
    for model, mape in mape_scores.items():
        if model != winner:
            improvement = ((mape - mape_scores[winner]) / mape) * 100
            print(f"   {model}: {mape:.2f}% ({improvement:.1f}% worse than winner)")
    
    # ============================================================
    # SAVE RESULTS TO DATABASE
    # ============================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS TO DATABASE")
    print("=" * 70)
    
    engine = create_engine("postgresql://admin:secret@db:5432/supply_chain")
    
    # Save comparison summary
    comparison.reset_index(inplace=True)
    comparison.rename(columns={'index': 'metric'}, inplace=True)
    comparison.to_sql('model_comparison', engine, if_exists='replace', index=False)
    print("✅ Saved model comparison to 'model_comparison' table")
    
    # Save detailed results per window
    all_results = []
    for model, res in results.items():
        res = res.copy()
        res['model'] = model
        all_results.append(res)
    
    detailed = pd.concat(all_results, ignore_index=True)
    detailed.to_sql('cv_results', engine, if_exists='replace', index=False)
    print("✅ Saved detailed CV results to 'cv_results' table")
    
    # Save feature importance (XGBoost only)
    xgb_importance = xgb_model.get_feature_importance()
    if xgb_importance is not None:
        xgb_importance.to_sql('feature_importance', engine, if_exists='replace', index=False)
        print("✅ Saved XGBoost feature importance to 'feature_importance' table")
    
    print("\n💾 All results saved! Query them with:")
    print("   SELECT * FROM model_comparison;")
    print("   SELECT * FROM cv_results;")
    
    return comparison, results

if __name__ == "__main__":
    run_all_models()