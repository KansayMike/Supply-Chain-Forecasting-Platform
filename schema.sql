-- ============================================================
-- TABLE 1: products
-- ============================================================
CREATE TABLE IF NOT EXISTS products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    unit_cost NUMERIC(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- TABLE 2: stores
-- ============================================================
CREATE TABLE IF NOT EXISTS stores (
    store_id SERIAL PRIMARY KEY,
    store_name VARCHAR(255) NOT NULL,
    city VARCHAR(100),
    country VARCHAR(100),
    store_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- TABLE 3: sales_transactions
-- ============================================================
CREATE TABLE IF NOT EXISTS sales_transactions (
    transaction_id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL,
    store_id INTEGER NOT NULL,
    transaction_date DATE NOT NULL,
    units_sold INTEGER NOT NULL CHECK (units_sold >= 0),  -- changed here
    unit_price NUMERIC(10,2) NOT NULL,
    total_revenue NUMERIC(12,2) GENERATED ALWAYS AS ((units_sold * unit_price)::NUMERIC(12,2)) STORED,
    CONSTRAINT fk_product FOREIGN KEY (product_id) REFERENCES products(product_id),
    CONSTRAINT fk_store FOREIGN KEY (store_id) REFERENCES stores(store_id)
);


-- ============================================================
-- TABLE 4: forecasts
-- ============================================================
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL,
    store_id INTEGER NOT NULL,
    forecast_date DATE NOT NULL,
    forecast_horizon INTEGER NOT NULL,
    predicted_units NUMERIC(10,2),
    model_name VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_forecast_product FOREIGN KEY (product_id) REFERENCES products(product_id),
    CONSTRAINT fk_forecast_store FOREIGN KEY (store_id) REFERENCES stores(store_id)
);

-- ============================================================
-- INDEXES
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_sales_date ON sales_transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_sales_product ON sales_transactions(product_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_lookup ON forecasts(product_id, store_id, forecast_date, model_name);

-- ============================================================
-- CONSTRAINT FIXES
-- ============================================================
ALTER TABLE sales_transactions
DROP CONSTRAINT sales_transactions_units_sold_check;

ALTER TABLE sales_transactions
ADD CONSTRAINT sales_transactions_units_sold_check CHECK (units_sold >= 0);
