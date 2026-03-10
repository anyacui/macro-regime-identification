from fredapi import Fred
import yfinance as yf
import pandas as pd

import os
from dotenv import load_dotenv
from config import FRED_SERIES, YAHOO_SERIES, START_DATE, ASSET_CLASSES
load_dotenv()

fred = Fred(api_key=os.getenv('FRED_API_KEY'))


def load_fred_series():
    series = {}

    # making dictionary of series of data
    for name, key in FRED_SERIES.items():
        series[name] = fred.get_series(key)
    return series


def load_yahoo_series(start_date=START_DATE):
    series = {}

    for name, key in YAHOO_SERIES.items():
        raw = yf.download(key, start=start_date)
        raw.columns = raw.columns.get_level_values(0)
        raw.index = raw.index.astype('datetime64[us]')
        raw.index.name = None
        series[name] = raw['Close']

    return series


def load_all():
    fred_data = load_fred_series()
    yahoo_data = load_yahoo_series()

    series_dict = fred_data | yahoo_data

    return series_dict


def load_asset_classes(df, start_date=START_DATE):
    series = {}

    for name, key in ASSET_CLASSES.items():
        raw = yf.download(key, start=start_date)
        raw.columns = raw.columns.get_level_values(0)
        raw.index = raw.index.astype('datetime64[us]')
        raw.index.name = None
        series[name] = raw['Close'].resample('ME').last().pct_change()

    asset_returns = pd.DataFrame({
        'equities': series['sp500'],
        'bonds': series['bonds'],
        'gold': series['gold'],
        'high_yield': series['high_yield'],
        'dollar': series['dollar']
    })

    # Align with regime labels
    asset_returns['regime'] = df['regime_label']
    # drop where regime has not been calculated (equities series starts 1 yr earlier)
    asset_returns = asset_returns.dropna(subset=['regime'])

    return asset_returns


def load_tbill(start_date=START_DATE):
    tbill = fred.get_series('TB3MS', observation_start=start_date)
    tbill_monthly = tbill.resample('ME').last() / 100 / 12
    tbill_monthly.name = 'risk_free'
    return tbill_monthly


if __name__ == '__main__':
    data = load_all()
    for key in data:
        print(key)
