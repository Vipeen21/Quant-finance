import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport   

# 1. DOWNLOAD DATA
# We'll use Apple (AAPL) data for the last 5 years
ticker = "AAPL"
data = yf.download(ticker, start="2019-01-01", end="2026-01-01")
print(data) #show 5 variables close, high, low,open and volume
print("It shows SMA50 data only after 50th row \n", data.head(60)) # will show sma50 only after 50th row
print("It shows SMA200 data only after 200th row \n", data.head(201)) #use data.tail(20) to see bottom data 
print("Summary of data\n",  data.describe())
print("shows technical summary of data structure")
data.info()

# Generates a comprehensive advance automated summaries in HTML report
#profile = ProfileReport(data, title="Data Summary Report")
#profile.to_file("summary of data - algo trading with backtesting.html")



# 2. CALCULATE INDICATORS
# Short-term (50-day) and Long-term (200-day) moving averages
data['SMA50'] = data['Close'].rolling(window=50).mean()
print(data['SMA50'])
data['SMA200'] = data['Close'].rolling(window=200).mean()
print(data['SMA200'])

# 3. GENERATE SIGNALS
# Signal = 1 when SMA50 > SMA200 (Long/Buy), else 0
data['Signal'] = 0
data.iloc[50:, data.columns.get_loc('Signal')] = (data['SMA50'][50:] > data['SMA200'][50:]
).astype(int) # we are locating the data starting from row 50 till end and finding the number of the 
# 'signal' column, since column is 8th column, we will use 7 for this as python starts counting at 0
print(data['Signal']) # i am here checking the signal column of the data starting 
#from row 51 to row 200
data['Signal'].iloc[195:205] #data.iloc[i:j] is used to locate between row data 

# 4. BACKTESTING (The logic)
# Position tells us if we just bought (1) or just sold (-1) or just hold (0)
data['Position'] = data['Signal'].diff()
data['Position'].iloc[195:205]
# Calculate daily returns of the stock
data['Market_Returns'] = data['Close'].pct_change()
data['Market_Returns'].iloc[195:205]

# Calculate strategy returns (Shift signal by 1 day because you trade at the NEXT open)
data['Strategy_Returns'] = data['Market_Returns'] * data['Signal'].shift(1)
data['Strategy_Returns'].iloc[195:205]

# Calculate Cumulative Returns
data['Cumulative_Market'] = (1 + data['Market_Returns']).cumprod() #.cumprod(): Cumulative Product. 
#It compounds the returns over time to show how $1 would have grown over the 5 years.
data['Cumulative_Market'].iloc[195:205]
data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).cumprod()
data['Cumulative_Strategy'].iloc[195:205]

# 5. VISUALIZATION
plt.figure(figsize=(12, 6))
plt.plot(data['Cumulative_Market'], label='Buy & Hold (Market)', color='gray', alpha=0.5)
plt.plot(data['Cumulative_Strategy'], label='SMA Crossover Strategy', color='blue')
plt.title(f'Backtest: {ticker} SMA Crossover Strategy')
plt.legend()
plt.show()

# Print Final Result
print(f"Final Market Return: {data['Cumulative_Market'].iloc[-1]:.2f}") #iloc[-1] means 
#the last item in the row
# final market return of 5.08 means, if you had put 1$ into Apple at 
# the start of the backtest and did nothing, you would now have 5.08$.
print(f"Final Strategy Return: {data['Cumulative_Strategy'].iloc[-1]:.2f}")
# similarly, final strategy return of 2.88 means, if you had put 1$ into Apple at 
# the start of the backtest and did nothing, you would now have 2.88$.
if data['Cumulative_Strategy'].iloc[-1] > data['Cumulative_Market'].iloc[-1]:
    print("Your algorithm outperformed buy and hold strategy")
else:
    print("Your algorithm is underperforming, and you would have been better off just holding the stock and forget.")


