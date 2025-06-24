#okay so this is going to be the new FamaFrench 3 factor model that I am working on let's see where it goes

import pandas_datareader.data as web
import matplotlib.pyplot as plt 
import streamlit as st
from datetime import datetime, timedelta
import datetime
import pandas as pd
import yfinance as yf
from statsmodels.api import OLS
import statsmodels.api as sm

st.header("Stress testing based on the previous code")
st.write("Here: 1. we specify the stress testing window, in this period we determine average monthly drawdown" \
" 2. Run a regression of the factors on the stock return for the past 5 years" \
" 3. Adjust the factors with the maximum average monthly drawdown to simulate the window scenario" \
" 4. Calculate the stock return based on the adjusted factors")
st.write("When we are comparing the Carhart 4 factor and the Fama-French 5-factor model, we should know these: 1.If the focus is on trading performance, return prediction, or momentum crashes → Stick with Carhart 4F or even combine it with 5F (a 6-Factor model!) 2. If we are analyzing fundamentals, corporate finance factors, or valuation-driven investing → Fama-French 5F may add more explanatory power.")

end_date = datetime.date.today()
start_date = (end_date - timedelta(days=5*365))
start_date_stress = st.sidebar.date_input("Input the starting date for retrieving the factor data for the stress testing window:", value="2007-01-01")
end_date_stress = st.sidebar.date_input("Input the ending date for the stress testing scenario:", value="2010-01-01")

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

ticker = st.sidebar.text_input("Specify which US stock you want to analyse (yahoofinance ticker):", value='AAPL')

# Fetch data with yfinance
ticker_df = yf.download(ticker, start=start_date, end=end_date, interval='1d')  # Daily data

# Resample to monthly adjusted close prices (month-end)
monthly_prices = ticker_df['Close'].resample('M').last()

# Calculate monthly percentage change
monthly_returns = monthly_prices.pct_change().dropna() * 100  # Multiply by 100 for %

st.write("monthly_returns", monthly_returns)

# Combine Series into DataFrame safely
stock_df = pd.concat([monthly_prices, monthly_returns], axis=1)
stock_df.columns = ['Adj Close', 'Monthly Return (%)']

# Display in Streamlit
st.write(f"Monthly Data for {ticker}", stock_df)

#this is a second way to achieve the same thing
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

# Add constant term to factors (this is the alpha!)
X = sm.add_constant(stock_carhart_merged_df[['Mkt-RF', 'SMB', 'HML', 'Mom']])
y = stock_carhart_merged_df[ticker]

# Run regression
model_robust = sm.OLS(y, X, missing='drop').fit(cov_type='HC3')

st.subheader('OLS regression with constant and robust standard errors for the past 5 years')
st.write(model_robust.summary())

#########################################################################################################################################
#so now I have the regression results with the constant for the past 5 years 
#the next step is to determine the greatest monthly drawdown for the factors 

ff3_stress = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start = start_date_stress, end = end_date_stress)[0]
mom_stress = web.DataReader('F-F_Momentum_Factor', 'famafrench', start = start_date_stress, end = end_date_stress)[0]
carhart4_stress = ff3_stress.merge(mom_stress, left_index=True, right_index=True)
st.write("Carhart 4-factor Model Data:", carhart4_stress)

# Helper function: compute max drawdown for a Series
def compute_max_drawdown(series: pd.Series):
    s = series.sort_index()
    if isinstance(s.index, pd.PeriodIndex):
        s.index = s.index.to_timestamp()
    running_peak = s.cummax()
    drawdowns = s - running_peak
    trough_date = drawdowns.idxmin()
    trough_value = s.loc[trough_date]
    peak_value = running_peak.loc[:trough_date].max()
    peak_date = s.loc[:trough_date][s.loc[:trough_date] == peak_value].index.max()
    months_to_trough = (trough_date.to_period('M') - peak_date.to_period('M')).n 
    max_drawdown = peak_value - trough_value
    avg_monthly_max_dd = max_drawdown / months_to_trough

    return {
        'Max Drawdown (%)': max_drawdown,
        'Peak Date': peak_date.strftime('%Y-%m'),
        'Trough Date': trough_date.strftime('%Y-%m'),
        'Peak value': peak_value,
        'Trough value:': trough_value,
        'Months to Trough': months_to_trough,
        'Avg Monthly DD (%)': avg_monthly_max_dd
    }

# Apply to all factors
results = {}
for factor in ['Mkt-RF', 'SMB', 'HML', 'RF', 'Mom']:
    results[factor] = compute_max_drawdown(carhart4_stress[factor])
# Plotting the factors using pandas' plot() and matplotlib
fig3, ax = plt.subplots(carhart4_stress.shape[1], 1, figsize=(10, 12), sharex=True)

carhart4_stress.plot(subplots=True, ax=ax, legend=False, title="Carhart 4-Factor Model Time Series")

# Set axis labels for clarity
for a, col in zip(ax, carhart4_stress.columns):
    a.set_ylabel(col)

plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig3)

# Build DataFrame
dd_df = pd.DataFrame(results).T
st.subheader("Stress-Window Maximum Drawdown Summary per Factor")
st.write(dd_df)

#extract the regression coefficients
st.subheader("Extracted regression coefficients")
coef_dict = model_robust.params.to_dict()
st.write(coef_dict)

#extract the latest factor values for the stress test
st.subheader("Extracted latest factor values (%)")
latest_factors = carhart4.iloc[-1][['Mkt-RF', 'SMB', 'HML', 'RF', 'Mom']].to_dict()
st.write(latest_factors)

#extract the monthly average maximum drawdown throughout the period
st.subheader("Extracted monthly average maximum drawdown (%)")
avg_dd_dict = dd_df['Avg Monthly DD (%)'].to_dict()
st.write(avg_dd_dict)

#computing the stressed factor values that will be plugged into the regression equation
stressed_factors = {}
for fac in ['Mkt-RF', 'SMB', 'HML', 'RF', 'Mom']:
    stressed_factors[fac] = latest_factors[fac] - avg_dd_dict[fac]
st.subheader("Adjusted stressed factors (subtraction)")    
st.write(stressed_factors)

# Compute the stressed predicted return (since RF is not in regression)
alpha = coef_dict['const']
predicted_stressed_return = alpha

for fac in ['Mkt-RF', 'SMB', 'HML', 'Mom']:
    predicted_stressed_return += coef_dict[fac] * stressed_factors[fac]

# Display the result
st.subheader(f"Predicted Stressed Monthly Return for {ticker}")
st.write(f"Predicted Stressed Return: {predicted_stressed_return:.3f}%")

st.subheader("Explanation")
st.write("A lower adjusted Mkt-RF value indicates that the market return is lower relative to the risk-free rate, meaning that in this stress scenario, there is a bear market, and there are market losses, flight to safety and panic selling, volatility etc. The stocks are down and the risk free assets are outperforming. relatively")
st.write("A lower SMB indicates that small caps are underperforming large cap stocks. There is a flight to safety, where investors avoid small, risky firms. This suggests, that small businesses are struggling, indicating a CRISIS PERIOD")
st.write("A lower HML means that value stocks are underperforming growth stocks. This indicates that growth stocks are relatively outperforming, meaning investors prefer high-earning firms. This reflects downturns like in the 2020 COVID rally, with defensiveness, with more established growth companies.")
st.write("This indicates a lower momentum, where the past winners underperform past losers. It reflects a reverse in the trend, where the market moves against recent trends, reflecting volatile and uncertain markets, where previous winners become losers. It is typical in market crashes or corrections.It is not necessarily a negative shock, just a great reversal from current values.")
st.write("In summary, lower factor values imply 1. broad market weakness 2. risk aversion 3. volatility in factor behaviour.")

st.subheader("For only the bear factors (all except momentum factor)")
bear_factors = ['Mkt-RF', 'SMB', 'HML']

stressed_factors_bear = {}
for fac in ['Mkt-RF', 'SMB', 'HML', 'RF', 'Mom']:
    if fac in bear_factors:
        stressed_factors_bear[fac] = latest_factors[fac] - avg_dd_dict[fac]
    else:
        stressed_factors_bear[fac] = latest_factors[fac]  # no change

st.write(stressed_factors_bear)

# Compute the predicted stressed return using bear factors
alpha = coef_dict['const']
predicted_stressed_return_bear = alpha

for fac in ['Mkt-RF', 'SMB', 'HML', 'Mom']:
    predicted_stressed_return_bear += coef_dict[fac] * stressed_factors_bear[fac]

# Display result
st.subheader(f"Predicted Stressed Monthly Return for {ticker} (Bear Market Factors Only)")
st.write(f"Predicted Stressed Return (Bear Factors): {predicted_stressed_return_bear:.3f}%")

# User-selectable number of months for the stress scenario
N_stress_months = st.sidebar.slider("Select stress scenario duration (months)", min_value=1, max_value=12, value=6)

# Compute stressed factors for bear market (N-month shock)
st.subheader(f"Adjusted stressed factors for Bear Market scenario ({N_stress_months}-Month Stress)")

stressed_factors_bear2 = {}
for fac in ['Mkt-RF', 'SMB', 'HML', 'Mom']:
    if fac in bear_factors:
        # Apply average monthly drawdown * N months
        stressed_factors_bear2[fac] = latest_factors[fac] - avg_dd_dict[fac] * N_stress_months
    else:
        stressed_factors_bear2[fac] = latest_factors[fac]  # no change for non-bear factors

st.write(stressed_factors_bear2)

predicted_stressed_return_bear2 = alpha
for fac in ['Mkt-RF', 'SMB', 'HML', 'Mom']:
    predicted_stressed_return_bear2 += coef_dict[fac] * stressed_factors_bear2[fac]
# Display result
st.subheader(f"Predicted Stressed Monthly Return for {ticker} (Bear Market Factors, {N_stress_months}-Month Adjustment)")
st.write(f"Predicted Stressed Return: {predicted_stressed_return_bear2:.3f}%")

st.write("But this will be slightly unrealistic because the momentum would change as well, so that now we adjust all of the factors:")

# Compute stressed factors for bear market (N-month shock)
st.subheader(f"Adjusted stressed factors for Bear Market scenario ({N_stress_months}-Month Stress)")

stressed_factors_bear3 = {}
for fac in ['Mkt-RF', 'SMB', 'HML', 'Mom']:
    stressed_factors_bear3[fac] = latest_factors[fac] - avg_dd_dict[fac] * N_stress_months

st.write(stressed_factors_bear3)

predicted_stressed_return_bear3 = alpha
for fac in ['Mkt-RF', 'SMB', 'HML', 'Mom']:
    predicted_stressed_return_bear3 += coef_dict[fac] * stressed_factors_bear3[fac]

# Display result
st.subheader(f"Predicted Stressed Monthly Return for {ticker} (Bear Market Factors, {N_stress_months}-Month Adjustment)")
st.write(f"Predicted Stressed Return: {predicted_stressed_return_bear3:.3f}%")
