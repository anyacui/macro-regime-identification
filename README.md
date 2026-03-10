# Macro Regime Identification & Cross-Asset Analysis

![S&P 500 with Macro Regimes](notebooks/images/regime_graph.png)

K-means clustering on 35 years of U.S. macroeconomic data to identify latent macro regimes — and test whether regime awareness improves portfolio construction.

---

## Key Findings

- **Four distinct regimes** identified empirically: Goldilocks, Easy Policy/Low Growth, Crisis, and Overheating, each with economically interpretable characteristics consistent with macro theory
- **Regime structure is robust** — Adjusted Rand Score of 1.0 against PCA-based clustering confirms regimes reflect genuine macro structure rather than feature correlation artifacts
- **Cross-asset Sharpe ratios vary significantly across regimes** — gold dominates in Crisis (0.82) and Goldilocks (1.27); dollar is the only positive-Sharpe asset in Overheating (0.83); bonds produce their worst risk-adjusted returns during Overheating (-0.95)
- **Regime-switching portfolio produces Sharpe of 1.08 vs 0.60 for static 60/40**, driven by lower volatility (7.5% vs 9.9%) and superior drawdown protection (-10.5% vs -30.0% maximum drawdown)

---

## Methodology

### Data
All macroeconomic data sourced from the FRED API. Market data from Yahoo Finance. Dataset spans January 1990 to present at monthly frequency.

### Features
Eight features engineered to span the key dimensions of the macro cycle:

| Feature | Source | Transformation |
|---|---|---|
| Real Fed Funds Rate | FRED | Fed Funds minus CPI YoY |
| Volatility Risk Premium | FRED + Yahoo | VIX minus annualised realised vol |
| DXY Year-on-Year | Yahoo | Pct change vs 12 months prior |
| Treasury Spread | FRED | 10y minus 2y yield |
| Credit Spread | FRED | BAA minus 10y yield |
| Unemployment Rate | FRED | Level |
| Industrial Production YoY | FRED | Pct change vs 12 months prior |
| CPI Year-on-Year | FRED | Pct change vs 12 months prior |

### Clustering
K-means clustering (k=4) on standardised features. Optimal k selected via elbow method and silhouette score, both converging on k=4.

### Robustness Check
PCA robustness check confirms clustering is not an artifact of feature correlation by re-running k-means on 6 PCA components (explaining 95% of variance) produces identical regime assignments (ARI = 1.0).

### Backtest
Sharpe-weighted regime-switching portfolio with one-month implementation lag to eliminate lookahead bias. Benchmarked against static 60/40 portfolio.

---

## Repository Structure

```
macro-regime-identification/
├── config.py               # Series codes, parameters, regime labels
├── main.py                 # Runs full pipeline end to end
├── src/
│   ├── data_loader.py      # FRED + Yahoo Finance data ingestion
│   ├── features.py         # Feature engineering and scaling
│   ├── clustering.py       # K-means, elbow, silhouette
│   └── backtest.py         # Asset performance, allocations, backtest
│   └── pca_analysis.py     # PCA explained variance, PCA clustering
└── notebooks/
    └── analysis.ipynb      # Full analysis with commentary
```

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/anyacui/macro-regime-identification.git
cd macro-regime-identification
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your FRED API key**

Create a `.env` file in the root directory:
```
FRED_API_KEY=your_key_here
```

Get a free API key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).

**5. Run the pipeline**
```bash
python main.py
```

Or open `notebooks/analysis.ipynb` for the full analysis with commentary.

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
fredapi
yfinance
python-dotenv
```

---

## Extensions

- **GMM** — probabilistic regime assignments to handle transition periods
- **Regime transition matrix** — probability of moving between regimes
- **Walk-forward validation** — out-of-sample regime detection
- **Portfolio optimisation** — mean-variance or risk parity allocations within each regime