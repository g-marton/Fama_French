#okay so this is going to be the new FamaFrench 3 factor model that I am working on let's see where it goes

import pandas_datareader.data as web
import matplotlib.pyplot as plt 
import streamlit as st
import datetime
import pandas as pd
import yfinance as yf
from statsmodels.api import OLS
import statsmodels.api as sm

start_date = st.sidebar.date_input("Input the starting date for retrieving the factor data:", value="2020-01-01")
end_date = datetime.date.today()

ff3 = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start = start_date, end = end_date)[0]
mom = web.DataReader('F-F_Momentum_Factor', 'famafrench', start = start_date, end = end_date)[0]
carhart4 = ff3.merge(mom, left_index=True, right_index=True)

#these will give us monthly data for the factors, and the research wisdom says that daily data is way too noisy, so that it is less likely to get a good measure of factor sensitivities, but monthly is better 
#the market return will be the market cap weighted returns on the NASDAQ, NYSE, AMEX and the risk free rate is the one-month treasury bill yield

st.write("Carhart 4-factor Model Data:", carhart4)

# Plotting the factors using pandas' plot() and matplotlib
fig1, ax = plt.subplots(carhart4.shape[1], 1, figsize=(10, 12), sharex=True)

carhart4.plot(subplots=True, ax=ax, legend=False, title="Carhart 4-Factor Model Time Series")

# Set axis labels for clarity
for a, col in zip(ax, carhart4.columns):
    a.set_ylabel(col)

plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig1)

# Convert PeriodIndex to TimestampIndex
carhart4.index = carhart4.index.to_timestamp()

# Calculate rolling average
rolling_window = st.sidebar.selectbox("Select window (in months):", [1, 3, 6, 12, 24, 36])
carhart4_rolling = carhart4.rolling(rolling_window).mean()

# Plot both raw factors and rolling average
fig2, ax = plt.subplots(carhart4.shape[1], 1, figsize=(12, 14), sharex=True)

for i, col in enumerate(carhart4.columns):
    ax[i].plot(carhart4.index, carhart4[col], label=f"{col} (Monthly)", alpha=0.5)
    ax[i].plot(carhart4_rolling.index, carhart4_rolling[col], label=f"{col} ({rolling_window}-Month MA)", linewidth=2)
    ax[i].set_ylabel(col)
    ax[i].legend()

plt.tight_layout()

# Display in Streamlit
st.pyplot(fig2)

ticker = st.sidebar.text_input("Specify which US stock you want to analyse (yahoofinance ticker):", value='AAPL')

# Fetch data with yfinance
ticker_df = yf.download(ticker, start=start_date, end=end_date, interval='1d')  # Daily data

# Resample to monthly adjusted close prices (month-end)
monthly_prices = ticker_df['Close'].resample('M').last()

# Calculate monthly percentage change
monthly_returns = monthly_prices.pct_change().dropna() * 100  # Multiply by 100 for %

st.write("monthly_prices", monthly_prices.head())
st.write("monthly_returns", monthly_returns.head())

# Combine Series into DataFrame safely
stock_df = pd.concat([monthly_prices, monthly_returns], axis=1)
stock_df.columns = ['Adj Close', 'Monthly Return (%)']

# Display in Streamlit
st.write(f"Monthly Data for {ticker}", stock_df)

#here is a first option to trim the databases so that the time windows align
carhart4_trim = carhart4.iloc[1:]
monthly_returns_trim = monthly_returns.iloc[:-2]

# Reset indices so they align by row position
carhart4_trim = carhart4_trim.reset_index(drop=True)
monthly_returns_trim = monthly_returns_trim.reset_index(drop=True)

# Merge row by row
combined_df = pd.concat([monthly_returns_trim, carhart4_trim], axis=1)

# Display result
#st.write("Aligned Data for Regression", combined_df)


monthly_returns['str_date'] = monthly_returns.index.astype(str)
monthly_returns['dt_date'] = pd.to_datetime(monthly_returns['str_date']).dt.strftime('%Y-%m')
#st.write(monthly_returns)

carhart4['str_date'] = carhart4.index.astype(str)
carhart4['dt_date'] = pd.to_datetime(carhart4['str_date']).dt.strftime('%Y-%m')
#st.write(carhart4)

stock_carhart_merged_df = pd.merge(monthly_returns, carhart4, how='inner', on='dt_date', sort=True, copy=True, indicator=False, validate='one_to_one')
stock_carhart_merged_df.drop(columns=['str_date_x', 'str_date_y'], inplace=True)
stock_carhart_merged_df[f'{ticker}-RF'] = stock_carhart_merged_df[ticker] - stock_carhart_merged_df['RF'] 
st.write(stock_carhart_merged_df)

results_noconstant = OLS(stock_carhart_merged_df[f'{ticker}-RF'], stock_carhart_merged_df[['Mkt-RF', 'SMB', 'HML', 'Mom']],missing='drop').fit()
st.subheader('OLS regression without constant and no robust standard errors')
st.write(results_noconstant.summary())

# Add constant term to factors (this is the alpha!)
X = sm.add_constant(stock_carhart_merged_df[['Mkt-RF', 'SMB', 'HML', 'Mom']])
y = stock_carhart_merged_df[f'{ticker}-RF']

# Run regression
model1 = sm.OLS(y, X, missing='drop').fit()
model_robust = sm.OLS(y, X, missing='drop').fit(cov_type='HC3')

# Display summary
st.subheader('OLS regression with constant no robust standard errors')
st.write(model1.summary())

st.subheader('OLS regression with constant and robust standard errors')
st.write(model_robust.summary())
