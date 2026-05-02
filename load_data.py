# load_data.py
# This script loads the Kaggle store sales data into our PostgreSQL database
# Run it with: docker exec -it supply_chain_python python load_data.py

import pandas as pd
from sqlalchemy import create_engine
import os

def load_data(csv_path='train.csv'):
    """
    Main function that orchestrates the entire loading process.
    Think of this as the director of a movie — it calls all the scenes in order.
    """
    
    print("🚀 Starting data load process...")
    
    # Step 1: Read CSV
    print(f"📖 Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df):,} rows")
    
    # Step 2: Clean
    print("🧹 Cleaning data...")
    df = clean_data(df)
    
    # Step 3: Transform
    print("🔄 Transforming to schema format...")
    products_df, stores_df, sales_df = transform_data(df)
    
    # Step 4: Load to database
    print("💾 Loading to database...")
    load_to_database(products_df, stores_df, sales_df)
    
    print("✅ Data load complete!")


def clean_data(df):
    """
    Takes raw DataFrame, returns cleaned DataFrame.
    This is where we fix all the problems we found during exploration.
    """
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Remove any potential duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    if removed > 0:
        print(f"   Removed {removed} duplicate rows")
    
    # Ensure no negative sales
    negative_count = (df['sales'] < 0).sum()
    if negative_count > 0:
        print(f"   Found {negative_count} negative sales — investigating")
        # In a real project, you'd investigate why these exist
        # For now, we'll keep them (could be returns)
    
    return df


def transform_data(df):
    """
    Converts the Kaggle format into our 3-table schema format.
    Returns three DataFrames: products, stores, sales_transactions
    """
    
    # PRODUCTS
    products_df = pd.DataFrame({
        'product_name': df['family'].unique(),
        'category': df['family'].unique(),
        'unit_cost': 0.0
    })
    
    # STORES
    stores_df = pd.DataFrame({
        'store_name': [f"Store_{n}" for n in sorted(df['store_nbr'].unique())],
        'city': 'Unknown',
        'country': 'Ecuador',
        'store_type': 'supermarket'
    })
    
    # Create ID mappings
    # We use sorted() to ensure consistent ordering
    families = sorted(df['family'].unique())
    store_numbers = sorted(df['store_nbr'].unique())
    
    family_to_id = {family: idx + 1 for idx, family in enumerate(families)}
    store_nbr_to_id = {nbr: idx + 1 for idx, nbr in enumerate(store_numbers)}
    
    # SALES
    sales_df = df.copy()
    sales_df['product_id'] = sales_df['family'].map(family_to_id)
    sales_df['store_id'] = sales_df['store_nbr'].map(store_nbr_to_id)
    
    sales_df = sales_df[['product_id', 'store_id', 'date', 'sales', 'onpromotion']].copy()
    sales_df.columns = ['product_id', 'store_id', 'transaction_date', 'units_sold', 'unit_price']
    sales_df['unit_price'] = 1.0  # Placeholder
    
    return products_df, stores_df, sales_df


def load_to_database(products_df, stores_df, sales_df):
    """
    Loads the three DataFrames into PostgreSQL.
    Uses the same connection logic we built in database.py
    """
    
    # Build connection string (same as database.py)
    connection_string = "postgresql://admin:secret@db:5432/supply_chain"
    engine = create_engine(connection_string)
    
    # Load in order: products and stores first (sales references them)
    print("   Loading products...")
    products_df.to_sql('products', engine, if_exists='append', index=False)
    
    print("   Loading stores...")
    stores_df.to_sql('stores', engine, if_exists='append', index=False)
    
    print("   Loading sales transactions... (this may take a minute)")
    sales_df.to_sql('sales_transactions', engine, if_exists='append', index=False)
    
    # Verify
    with engine.connect() as conn:
        from sqlalchemy import text
        for table in ['products', 'stores', 'sales_transactions']:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.fetchone()[0]
            print(f"   ✅ {table}: {count:,} rows")


# This runs only when you execute this file directly
if __name__ == "__main__":
    load_data()