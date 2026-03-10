from src.data_loader import load_all
from sklearn.preprocessing import StandardScaler
import pandas as pd


def engineer_features():

    series_dict = load_all()

    # All data points are taken as month end
    # CPI
    series_dict['cpi_yoy'] = series_dict['cpi'].pct_change(12)*100
    series_dict['cpi_yoy'] = series_dict['cpi_yoy'] .resample('ME').last()

    # Industrial Production
    indpro_yoy = series_dict['indust_prod'].pct_change(12) * 100
    indpro_yoy = indpro_yoy.resample('ME').last()

    series_dict['fed_funds'] = series_dict['fed_funds'].resample('ME').last()

    # Monthly real FED rate
    real_rate = series_dict['fed_funds'] - series_dict['cpi_yoy']
    real_rate = real_rate.resample('ME').last()

    # Unemployment Rate
    unemployment_rate = series_dict['unemployment_rate'].resample('ME').last()

    # Credit Spread
    credit_spread_monthly = series_dict['credit_spread'].resample('ME').last()

    # 10 yr 2yr treasury spread
    treasury_spread = series_dict['ten_yr_yield'] - series_dict['two_yr_yield']
    treasury_spread = treasury_spread.resample('ME').last()

    # Volatility risk premium
    # calculating realized volatility from SPX returns
    daily_returns = series_dict['sp500_daily'].pct_change()
    realized_vol = daily_returns.resample('ME').std() * (252**0.5)
    # transform expected volatility into monthly measure
    vix_monthly = series_dict['vix'].resample('ME').mean()
    # spread calculation
    vol_risk_premium = vix_monthly - realized_vol

    # Dollar strength
    dxy_monthly = series_dict['dollar_strength'].resample(
        'ME').last()
    dxy_yoy = dxy_monthly.pct_change(12) * 100

    # Loading into dataframe
    df = pd.DataFrame({
        'real_rate': real_rate,
        'vol_risk_premium': vol_risk_premium,
        'dxy_monthly': dxy_yoy,
        'treasury_spread': treasury_spread,
        'credit_spread': credit_spread_monthly,
        'unemployment_rate': unemployment_rate,
        'indust_prod': indpro_yoy,
        'cpi': series_dict['cpi_yoy']
    })

    df = df.dropna()

    return df


def standardise_features(df):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )

    # return tuple
    return df_scaled, scaler


if __name__ == '__main__':
    df = engineer_features()
    print(df.head())
    print(df.tail())
    df_scaled, scaler = standardise_features(df)

    print(df_scaled.head())
    print(df_scaled.describe())
