# inventory.py
# Inventory optimization: convert forecasts into actionable stock levels

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

def calculate_safety_stock(forecast_errors, lead_time_days=7, service_level=0.95):
    """
    Calculates safety stock using forecast uncertainty.
    """
    z_scores = {
        0.80: 0.84,
        0.85: 1.04,
        0.90: 1.28,
        0.95: 1.65,
        0.99: 2.33,
        0.999: 3.09
    }
    z = z_scores.get(service_level, 1.65)
    sigma = np.std(forecast_errors, ddof=1)  # sample std dev
    lt_factor = np.sqrt(lead_time_days)
    safety_stock = z * sigma * lt_factor
    return max(safety_stock, 0)


def calculate_reorder_point(forecasted_demand, safety_stock, lead_time_days=7):
    """
    Reorder Point = Expected demand during lead time + Safety stock
    """
    avg_daily_demand = np.mean(forecasted_demand)
    demand_during_lead_time = avg_daily_demand * lead_time_days
    return demand_during_lead_time + safety_stock


def calculate_inventory_metrics(df, forecast_col='predicted', actual_col='units_sold'):
    """
    Calculates comprehensive inventory metrics from forecast vs actual.
    """
    forecast_errors = df[actual_col] - df[forecast_col]
    return {
        'mean_demand': df[actual_col].mean(),
        'demand_std': df[actual_col].std(),
        'mean_forecast': df[forecast_col].mean(),
        'forecast_bias': forecast_errors.mean(),
        'mae': np.mean(np.abs(forecast_errors)),
        'rmse': np.sqrt(np.mean(forecast_errors ** 2)),
        'forecast_error_std': forecast_errors.std()
    }


def generate_inventory_recommendations(
    product_id=1,
    store_id=1,
    model_name='XGBoost',
    lead_time_days=7,
    service_level=0.95,
    forecast_horizon=30
):
    """
    Generates full inventory recommendations for a product-store combination.
    """
    print(f"📦 Generating inventory recommendations...")
    print(f"   Product: {product_id}, Store: {store_id}")
    print(f"   Model: {model_name}, Lead time: {lead_time_days} days")
    print(f"   Service level: {service_level * 100}%")

    engine = create_engine("postgresql://admin:secret@db:5432/supply_chain")

    # Step 1: Get actual sales history
    query_actual = f"""
    SELECT transaction_date, units_sold
    FROM sales_transactions
    WHERE product_id = {product_id} AND store_id = {store_id}
    ORDER BY transaction_date
    """
    actual_df = pd.read_sql(query_actual, engine)

    # Step 2: Get CV results
    query_cv = f"""
    SELECT test_start, test_end, mae, rmse, mape
    FROM cv_results
    WHERE model = '{model_name}'
    ORDER BY test_start
    """
    cv_df = pd.read_sql(query_cv, engine)

    if len(cv_df) == 0:
        print("❌ No CV results found. Run final_comparison.py first!")
        return None

    # Step 3: Use last 30 days of demand
    recent_demand = actual_df.tail(forecast_horizon)['units_sold'].values

    # Step 4: Forecast error std from CV results
    forecast_error_std = cv_df['rmse'].mean()
    np.random.seed(42)
    synthetic_errors = np.random.normal(0, forecast_error_std, 1000)

    safety_stock = calculate_safety_stock(
        synthetic_errors,
        lead_time_days=lead_time_days,
        service_level=service_level
    )

    reorder_point = calculate_reorder_point(
        recent_demand,
        safety_stock,
        lead_time_days
    )

    # Step 5: Simplified EOQ
    annual_demand = actual_df['units_sold'].mean() * 365
    order_quantity = np.sqrt(2 * annual_demand * 10 / 0.5)  # S=$10, H=$0.50

    recommendations = {
        'product_id': product_id,
        'store_id': store_id,
        'model_used': model_name,
        'lead_time_days': lead_time_days,
        'service_level': service_level,
        'forecast_error_std': forecast_error_std,
        'safety_stock_units': round(safety_stock, 2),
        'reorder_point_units': round(reorder_point, 2),
        'order_quantity_units': round(order_quantity, 2),
        'max_inventory_units': round(reorder_point + order_quantity, 2),
        'avg_daily_demand': round(recent_demand.mean(), 2),
        'generated_at': pd.Timestamp.now()
    }

    print(f"\n📊 Recommendations:")
    print(f"   Safety Stock:     {recommendations['safety_stock_units']} units")
    print(f"   Reorder Point:    {recommendations['reorder_point_units']} units")
    print(f"   Order Quantity:   {recommendations['order_quantity_units']} units")
    print(f"   Max Inventory:    {recommendations['max_inventory_units']} units")

    return recommendations


def save_recommendations_to_db(recommendations):
    """
    Saves inventory recommendations to the database.
    """
    if recommendations is None:
        return

    engine = create_engine("postgresql://admin:secret@db:5432/supply_chain")
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT INTO inventory_plan (
                product_id, store_id, model_used, lead_time_days, service_level,
                forecast_error_std, safety_stock_units, reorder_point_units,
                order_quantity_units, max_inventory_units, avg_daily_demand, generated_at
            ) VALUES (
                :product_id, :store_id, :model_used, :lead_time_days, :service_level,
                :forecast_error_std, :safety_stock_units, :reorder_point_units,
                :order_quantity_units, :max_inventory_units, :avg_daily_demand, :generated_at
            )
            """),
            recommendations
        )

    print("✅ Recommendations saved to database")


if __name__ == "__main__":
    recs = generate_inventory_recommendations(
        product_id=1,
        store_id=1,
        model_name='XGBoost',
        lead_time_days=7,
        service_level=0.95
    )
    save_recommendations_to_db(recs)
