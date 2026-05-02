# Supply Chain Optimization with Time Series Forecasting

&gt; **Production-ready demand forecasting system comparing ARIMA, Prophet, and XGBoost with rolling cross-validation, dynamic safety stock optimization, and an interactive stakeholder dashboard.**

## Why This Project Exists

Retail supply chains lose millions to two opposing failures:
- **Stockouts** → Lost revenue, unhappy customers
- **Overstocking** → Capital lockup, spoilage, storage costs

This system solves both by forecasting demand with multiple models, rigorously comparing them, and automatically calculating optimal inventory levels.


| What I Built | Why It Matters |
|-------------|---------------|
| **3-model comparison** (ARIMA, Prophet, XGBoost) | I evaluate models, not just build them |
| **Rolling cross-validation** | I understand that time series can't be shuffled |
| **Statistical significance testing** | I don't claim victory without proof |
| **Dynamic safety stock** | I connect forecasts to business decisions |
| **SQL schema design** | I think in data architecture, not just notebooks |
| **Docker containerization** | I build production-ready systems, not prototypes |
| **Interactive dashboard** | I communicate with non-technical stakeholders |


## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data | PostgreSQL | Relational storage with proper schema |
| ETL | Python, Pandas, SQLAlchemy | Data loading and cleaning |
| Models | ARIMA (pmdarima), Prophet, XGBoost | Multi-horizon forecasting |
| Validation | Scikit-learn, SciPy | Metrics and statistical testing |
| Inventory | Custom Python module | Safety stock & reorder points |
| Deployment | Docker, Docker Compose | Reproducible environment |
| Dashboard | Streamlit, Plotly | Stakeholder visualization |

## Quick Start

```bash
# 1. Clone and enter directory
cd supply-chain-forecast

# 2. Download Kaggle data
# Place train.csv in this folder
# https://www.kaggle.com/c/store-sales-time-series-forecasting

# 3. Start everything
docker-compose up -d

# 4. Run the pipeline
docker-compose run pipeline

# 5. View dashboard
open http://localhost:8501

├── docker-compose.yml      # Infrastructure orchestration
├── Dockerfile              # Python environment
├── requirements.txt        # Dependencies
├── schema.sql              # Database design
├── database.py             # Connection helper
├── load_data.py            # ETL pipeline
├── features.py             # Feature engineering (XGBoost)
├── evaluation.py           # Rolling CV framework
├── inventory.py            # Safety stock calculator
├── pipeline.py             # End-to-end orchestration
├── dashboard.py            # Streamlit dashboard
├── models/
│   ├── arima_model.py      # Statistical baseline
│   ├── prophet_model.py    # Trend + seasonality + holidays
│   └── xgboost_model.py    # Gradient boosting
└── README.md               # This file

