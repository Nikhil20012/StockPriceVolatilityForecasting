# Stock Price and Volatility Forecasting

Stock price and volatility forecasting using statistical and deep learning models on NVDA, AAPL, and TSLA daily data (2010-2023). Compares Naive, Moving Average, ARIMA, GARCH(1,1), and GRU across multiple evaluation horizons.

Built for IE 7275 Data Mining Engineering at Northeastern University, Spring 2026.

## Problem

Given historical daily stock price data (Open, High, Low, Close, Adjusted Close, Volume) for NVIDIA, Apple, and Tesla, can we build models that accurately forecast future prices and volatility? The project addresses four problems:

1. **Price Level Forecasting** - Predict adjusted close price for each day in the test set (Jan 2021 - Jan 2023)
2. **Volatility Forecasting** - Forecast 20-day annualized realized volatility using GARCH(1,1)
3. **Model Comparison** - Rank all models by MAE, RMSE, and MAPE
4. **Horizon Analysis** - Measure how accuracy degrades over short (30-day) vs long (~480-day) horizons

## Dataset

- **Source:** [Big Tech Giants Stock Price Dataset](https://www.kaggle.com/datasets/umerhaddii/big-tech-giants-stock-price-dataset) (Kaggle, by Umer Haddii)
- **Tickers used:** NVDA, AAPL, TSLA
- **Date range:** 2010-01-04 to 2023-01-24
- **Total rows after filtering:** 9,690 (10,040 after calendar reindex)
- **Train/Val/Test split:** 70% / 15% / 15%, strictly chronological

## Preprocessing

- Parsed dates, enforced float64 types on all OHLCV columns
- Zero missing values, zero duplicates, zero OHLC constraint violations
- Outlier detection via z-score on daily log returns (299 rows with |z| > 4, all genuine market events)
- Winsorization at 1st/99th percentile per ticker instead of removal
- Calendar reindex to full business-day range with forward-fill

## Feature Engineering

| Feature | Description |
|---------|-------------|
| log_return | ln(P_t / P_{t-1}), makes returns stationary |
| sma_5, sma_10, sma_20 | Rolling mean of adjusted close (short/medium/long trend) |
| std_5, std_10, std_20 | Rolling std of adjusted close (price dispersion) |
| realized_vol_20 | 20-day rolling std of log returns, annualized (x sqrt(252)) |
| price_range | High - Low per day (intraday volatility proxy) |

**Normalization:**
- Min-Max Scaling (0-1) for price-level features (used as GRU inputs)
- Standard Scaling (zero mean, unit variance) for return/volatility features
- Scalers fit only on training set, then applied to val/test (no data leakage)

## Models

### Baseline Models

- **Naive Forecast:** Tomorrow's price = today's price. Based on the random walk hypothesis.
- **Moving Average (MA-20):** Constant prediction using mean of last 20 training prices.

### ARIMA

- Grid search over p in {0,1,2,3}, d in {0,1}, q in {0,1,2,3} (32 combinations per ticker)
- Best order selected by minimizing AIC
- All three tickers required d=1 (first-order differencing)
- Best orders: NVDA (3,1,3), AAPL (3,1,3), TSLA (2,1,2)

### GARCH(1,1)

- Models time-varying volatility where today's variance depends on yesterday's squared return and yesterday's variance
- Log returns scaled x100 for numerical stability
- Fitted using the `arch` library
- Evaluated against 20-day realized volatility

### GRU (Gated Recurrent Unit)

- Two-layer GRU built with TensorFlow/Keras
- Sliding window of 30 days as input, next day's price as target
- Trained separately per ticker

| Layer | Details |
|-------|---------|
| Input | (30, 1) |
| GRU Layer 1 | 64 units, return_sequences=True |
| Dropout 1 | 0.20 |
| GRU Layer 2 | 32 units |
| Dropout 2 | 0.20 |
| Dense Output | 1 unit, linear activation |

Training: Adam optimizer, MSE loss, batch size 32, max 100 epochs, early stopping (patience=10).

## Results

### Price Forecasting (Full Test Horizon)

| Model | NVDA MAE | NVDA MAPE | AAPL MAE | AAPL MAPE | TSLA MAE | TSLA MAPE |
|-------|----------|-----------|----------|-----------|----------|-----------|
| Naive | 4.43 | 2.40% | 1.96 | 1.34% | 6.37 | 2.54% |
| GRU | 6.72 | 3.41% | 3.21 | 2.13% | 8.91 | 3.47% |
| ARIMA | 153.39 | 79.13% | 105.30 | 71.17% | 241.03 | 92.46% |
| Moving Avg | 153.84 | 79.38% | 109.20 | 73.84% | 241.22 | 92.54% |

### Volatility Forecasting (GARCH)

| Ticker | MAE | RMSE | MAPE |
|--------|-----|------|------|
| NVDA | 0.15 | 0.17 | 51.01% |
| AAPL | 0.08 | 0.10 | 29.95% |
| TSLA | 0.14 | 0.18 | 30.36% |

*GARCH MAE/RMSE are in annualized volatility units, not USD.*

### GRU Short vs Long Horizon

| Ticker | Short (30d) MAPE | Long (~460-479d) MAPE |
|--------|------------------|-----------------------|
| NVDA | 2.84% | 3.45% |
| AAPL | 1.98% | 2.14% |
| TSLA | 2.82% | 3.51% |

## Key Findings

- **GRU is the best learned model**, achieving 2.13-3.47% MAPE across all tickers. It correctly captures directional trends and regime shifts.
- **Naive wins on raw MAPE** (1.34-2.54%) but provides zero forward-looking signal. It simply echoes the last observed price.
- **ARIMA is unsuitable for long-horizon forecasting** without daily rolling refit. Its single-pass forecast converges to a near-constant value within a few steps, producing 71-92% MAPE.
- **GARCH(1,1) correctly identifies volatility regimes** (low vol in early 2021, high vol through 2022) but underestimates spike magnitudes during sudden market dislocations.
- **Short-horizon GRU consistently outperforms long-horizon** by ~0.5-0.6 percentage points, confirming that forecast uncertainty compounds over time.

## Limitations

- GRU used only a single feature (normalized adjusted close) despite 35 engineered columns being available
- ARIMA and GARCH were fit once on training data without rolling refit
- No exogenous variables (macro indicators, sentiment, sector indices)
- No hyperparameter tuning for GRU (fixed architecture)
- LSTM was not implemented for comparison
- No transaction cost modeling or backtesting

## Future Work

- Rolling/online ARIMA and GARCH refit
- Multi-feature GRU/LSTM with volume, volatility, and macro indicators
- EGARCH/GJR-GARCH for asymmetric volatility modeling
- Transformer-based models (Temporal Fusion Transformer, Informer)
- Ensemble/hybrid models combining GRU price forecasts with GARCH confidence intervals
- Hyperparameter optimization using Optuna
- Sentiment analysis integration using FinBERT
- Walk-forward cross-validation
- Directional accuracy evaluation

## Tech Stack

- **Language:** Python
- **ML/DL:** TensorFlow, Keras, scikit-learn, statsmodels, arch
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Data source:** Kaggle (Big Tech Giants Stock Price Dataset)

## Repository Structure

```
StockPriceVolatilityForecasting/
├── README.md
├── requirements.txt
└── notebooks/
    └── stock_price_volatility_forecasting.ipynb
```

## Setup

1. Clone the repo
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/umerhaddii/big-tech-giants-stock-price-dataset)
3. Install dependencies: `pip install -r requirements.txt`
4. Open the notebook in Google Colab or Jupyter and upload the CSV files when prompted
