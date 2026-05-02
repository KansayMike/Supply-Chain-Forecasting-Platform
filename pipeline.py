# pipeline.py
# Production pipeline: end-to-end orchestration

import sys
import logging
import traceback
from datetime import datetime
from sqlalchemy import create_engine, text
import pandas as pd

# Import our modules
from database import test_connection
from load_data import load_data
from evaluation import RollingCV
from models.arima_model import ARIMAModel
from models.prophet_model import ProphetModel
from models.xgboost_model import XGBoostModel
from inventory import generate_inventory_recommendations, save_recommendations_to_db

# ============================================================
# LOGGING SETUP
# ============================================================
# Logging is like a flight recorder for your code.
# When something breaks at 3 AM, logs tell you what happened.

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Print to console
        logging.FileHandler('pipeline.log')  # Save to file
    ]
)
logger = logging.getLogger(__name__)


class Pipeline:
    """
    The Pipeline class orchestrates the entire forecasting workflow.
    
    Think of it like an assembly line in a factory:
    1. Check raw materials (database connection)
    2. Load ingredients (data ingestion)
    3. Build products (train models)
    4. Quality check (cross-validation)
    5. Package for shipping (inventory recommendations)
    """
    
    def __init__(self, product_id=1, store_id=1):
        self.product_id = product_id
        self.store_id = store_id
        self.results = {}
        self.start_time = datetime.now()
        
    def step_1_verify_database(self):
        """Step 1: Verify database is accessible."""
        logger.info("=" * 60)
        logger.info("STEP 1: Verifying database connection")
        logger.info("=" * 60)
        
        if not test_connection():
            raise ConnectionError("Cannot connect to database. Aborting.")
        
        logger.info("✅ Database connection verified")
        return True
    
    def step_2_load_data(self, csv_path='train.csv'):
        """Step 2: Load raw data if tables are empty."""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Checking data availability")
        logger.info("=" * 60)
        
        engine = create_engine("postgresql://admin:secret@db:5432/supply_chain")
        
        # Check if data exists
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM sales_transactions"))
            count = result.fetchone()[0]
        
        if count == 0:
            logger.info(f"Database empty. Loading data from {csv_path}...")
            load_data(csv_path)
        else:
            logger.info(f"Database already has {count:,} sales transactions. Skipping load.")
        
        logger.info("✅ Data available")
        return True
    
    def step_3_train_and_evaluate(self):
        """Step 3: Train all models and evaluate with rolling CV."""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Training and evaluating models")
        logger.info("=" * 60)
        
        # Fetch data
        engine = create_engine("postgresql://admin:secret@db:5432/supply_chain")
        query = f"""
        SELECT transaction_date, units_sold
        FROM sales_transactions
        WHERE product_id = {self.product_id} AND store_id = {self.store_id}
        ORDER BY transaction_date
        """
        df = pd.read_sql(query, engine)
        
        if len(df) < 730:
            raise ValueError(f"Insufficient data: {len(df)} rows. Need at least 730.")
        
        logger.info(f"Loaded {len(df)} days of data")
        
        # Cross-validation setup
        cv = RollingCV(min_train_size=730, test_size=30, step=90)
        
        # Train all three models
        models = {
            'ARIMA': ARIMAModel(seasonal=False),
            'Prophet': ProphetModel(country='EC'),
            'XGBoost': XGBoostModel()
        }
        
        all_results = []
        
        for name, model in models.items():
            logger.info(f"\nTraining {name}...")
            try:
                results = cv.evaluate_model(model, df)
                results['model'] = name
                all_results.append(results)
                self.results[name] = results
                
                logger.info(f"✅ {name} completed successfully")
                
            except Exception as e:
                logger.error(f"❌ {name} failed: {str(e)}")
                logger.error(traceback.format_exc())
                # Continue with other models even if one fails
        
        if not all_results:
            raise RuntimeError("All models failed. Check logs.")
        
        # Save results to database
        detailed = pd.concat(all_results, ignore_index=True)
        detailed.to_sql('cv_results', engine, if_exists='replace', index=False)
        
        # Create summary comparison
        comparison = pd.DataFrame({
            model: {
                'MAE': res['mae'].mean(),
                'RMSE': res['rmse'].mean(),
                'MAPE': res['mape'].mean()
            }
            for model, res in self.results.items()
        })
        comparison.to_sql('model_comparison', engine, if_exists='replace', index=False)
        
        # Determine winner
        mape_scores = {m: r['mape'].mean() for m, r in self.results.items()}
        self.winner = min(mape_scores, key=mape_scores.get)
        
        logger.info(f"\n🏆 Winner: {self.winner} (MAPE: {mape_scores[self.winner]:.2f}%)")
        
        # Save feature importance if XGBoost ran
        if 'XGBoost' in models and hasattr(models['XGBoost'], 'get_feature_importance'):
            try:
                importance = models['XGBoost'].get_feature_importance()
                if importance is not None:
                    importance.to_sql('feature_importance', engine, if_exists='replace', index=False)
                    logger.info("✅ Feature importance saved")
            except:
                pass
        
        return True
    
    def step_4_generate_inventory(self, lead_time=7, service_level=0.95):
        """Step 4: Generate inventory recommendations using winner."""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Generating inventory recommendations")
        logger.info("=" * 60)
        
        recs = generate_inventory_recommendations(
            product_id=self.product_id,
            store_id=self.store_id,
            model_name=self.winner,
            lead_time_days=lead_time,
            service_level=service_level
        )
        
        if recs:
            save_recommendations_to_db(recs)
            logger.info("✅ Inventory recommendations saved")
        
        return True
    
    def step_5_generate_forecasts(self, days_ahead=30):
        """Step 5: Generate future forecasts and save to database."""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Generating future forecasts")
        logger.info("=" * 60)
        
        engine = create_engine("postgresql://admin:secret@db:5432/supply_chain")
        
        # Get latest data
        query = f"""
        SELECT transaction_date, units_sold
        FROM sales_transactions
        WHERE product_id = {self.product_id} AND store_id = {self.store_id}
        ORDER BY transaction_date
        """
        df = pd.read_sql(query, engine)
        
        # Train winner on full history
        if self.winner == 'XGBoost':
            model = XGBoostModel()
        elif self.winner == 'Prophet':
            model = ProphetModel(country='EC')
        else:
            model = ARIMAModel(seasonal=False)
        
        logger.info(f"Training {self.winner} on full history...")
        model.fit(df, date_col='transaction_date', target_col='units_sold')
        
        # Generate future dates
        last_date = df['transaction_date'].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=days_ahead,
            freq='D'
        )
        
        future_df = pd.DataFrame({
            'transaction_date': future_dates,
            'units_sold': 0  # Dummy for feature engineering
        })
        
        # Predict
        predictions = model.predict(future_df, date_col='transaction_date')
        
        # Save to forecasts table
        forecast_records = []
        for date, pred in predictions.items():
            forecast_records.append({
                'product_id': self.product_id,
                'store_id': self.store_id,
                'forecast_date': date,
                'forecast_horizon': days_ahead,
                'predicted_units': round(pred, 2),
                'model_name': self.winner,
                'created_at': datetime.now()
            })
        
        forecast_df = pd.DataFrame(forecast_records)
        forecast_df.to_sql('forecasts', engine, if_exists='append', index=False)
        
        logger.info(f"✅ Saved {len(forecast_df)} days of forecasts")
        
        return True
    
    def run(self):
        """Execute the full pipeline."""
        logger.info("🚀 PIPELINE STARTED")
        logger.info(f"Product: {self.product_id}, Store: {self.store_id}")
        logger.info(f"Start time: {self.start_time}")
        
        try:
            self.step_1_verify_database()
            self.step_2_load_data()
            self.step_3_train_and_evaluate()
            self.step_4_generate_inventory()
            self.step_5_generate_forecasts()
            
            elapsed = (datetime.now() - self.start_time).total_seconds()
            logger.info("\n" + "=" * 60)
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"⏱️  Total time: {elapsed:.1f} seconds")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error("\n" + "=" * 60)
            logger.error("❌ PIPELINE FAILED")
            logger.error(f"Error: {str(e)}")
            logger.error(traceback.format_exc())
            logger.error("=" * 60)
            return False


if __name__ == "__main__":
    # Run the pipeline
    pipeline = Pipeline(product_id=1, store_id=1)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)