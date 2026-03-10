import yfinance as yf
import pandas as pd
from config import ASSET_CLASSES, START_DATE
from src.features import engineer_features, standardise_features
from src.clustering import fit_clusters


def prepare_regime_data():
    df = engineer_features()
    df_scaled, scaler = standardise_features(df)
    df, kmeans = fit_clusters(df, df_scaled)

    return df, kmeans


def asset_regime_performance(asset_returns, tbill_monthly):
    results = {}

    for regime in asset_returns['regime'].unique():
        regime_data = asset_returns.loc[asset_returns['regime'] == regime].drop(
            columns='regime')

        rf_regime = tbill_monthly.reindex(regime_data.index)
        mean_annual = regime_data.mean() * 12
        vol_annual = regime_data.std() * (12**0.5)
        excess_return = regime_data.subtract(rf_regime, axis=0)
        sharpe = (excess_return.mean() * 12) / vol_annual
        corr = regime_data.corr()

        results[regime] = {
            'mean_return': mean_annual,
            'volatility': vol_annual,
            'sharpe': sharpe,
            'correlation': corr
        }

    return results

# sharpe-weighted allocations per regime


def compute_allocations(results):
    allocations = {}
    for regime in results:
        sharpe = results[regime]['sharpe']
        positive = sharpe[sharpe > 0]
        allocations[regime] = (positive / positive.sum()).to_dict()
    return allocations

# monthly portfolio returns on detected regime


def run_backtest(asset_returns, allocations, use_lag=True):
    # Apply lag if requested
    if use_lag:
        regime_col = asset_returns['regime'].shift(1)
    else:
        regime_col = asset_returns['regime']

    portfolio_returns = []

    for date, row in asset_returns.iterrows():
        regime = regime_col[date]
        if pd.isna(regime):
            continue

        alloc = allocations[regime]
        monthly_return = sum(
            alloc[asset] * row[asset]
            for asset in alloc
            if not pd.isna(row[asset])
        )
        portfolio_returns.append(
            {'date': date, 'regime_portfolio': monthly_return})

    portfolio_df = pd.DataFrame(portfolio_returns).set_index('date')

    # 60/40 benchmark
    portfolio_df['sixty_forty'] = (
        0.60 * asset_returns['equities'] +
        0.40 * asset_returns['bonds']
    )

    portfolio_df = portfolio_df.dropna()

    return portfolio_df


def compute_drawdown(returns_series):
    cumulative = (1 + returns_series).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max * 100
    return drawdown
