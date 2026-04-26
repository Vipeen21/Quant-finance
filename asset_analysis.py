import warnings
warnings.filterwarnings('ignore')

from openbb import obb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.signal import find_peaks

# Configure pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 4)

# Ask user for asset class
asset_class = input("Choose asset class (stocks, crypto, forex): ").strip().lower()

# Ask for asset name based on class
if asset_class == 'stocks':
    symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
    provider = "yfinance"
    data = obb.equity.price.historical(symbol=symbol, provider=provider, start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")).to_df()
elif asset_class == 'crypto':
    symbol = input("Enter crypto symbol (e.g., BTCUSD): ").strip().upper()
    provider = "yfinance"
    data = obb.crypto.price.historical(symbol=symbol, provider=provider, start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")).to_df()
elif asset_class == 'forex':
    symbol = input("Enter forex pair (e.g., EURUSD): ").strip().upper()
    provider = "yfinance"
    data = obb.currency.price.historical(symbol=symbol, provider=provider, start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")).to_df()
else:
    print("Invalid asset class. Defaulting to BTCUSD.")
    symbol = "BTCUSD"
    provider = "yfinance"
    data = obb.crypto.price.historical(symbol=symbol, provider=provider, start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")).to_df()

# ...existing code...

# Calculate advanced indicators
data['SMA_20'] = data['close'].rolling(20).mean()
data['SMA_50'] = data['close'].rolling(50).mean()
data['SMA_200'] = data['close'].rolling(200).mean()

# Bollinger Bands
data['BB_middle'] = data['close'].rolling(20).mean()
data['BB_std'] = data['close'].rolling(20).std()
data['BB_upper'] = data['BB_middle'] + (data['BB_std'] * 2)
data['BB_lower'] = data['BB_middle'] - (data['BB_std'] * 2)
data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']

# RSI
delta = data['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# MACD
exp1 = data['close'].ewm(span=12).mean()
exp2 = data['close'].ewm(span=26).mean()
data['MACD'] = exp1 - exp2
data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
data['MACD_hist'] = data['MACD'] - data['MACD_signal']

# ATR (Average True Range)
data['TR'] = np.maximum(
   data['high'] - data['low'],
   np.maximum(
       abs(data['high'] - data['close'].shift()),
       abs(data['low'] - data['close'].shift())
   )
)
data['ATR'] = data['TR'].rolling(14).mean()

# Identify support/resistance levels using peaks
peaks_high, _ = find_peaks(data['high'], distance=20, prominence=data['high'].std())
peaks_low, _ = find_peaks(-data['low'], distance=20, prominence=data['low'].std())

resistance_levels = data['high'].iloc[peaks_high].nlargest(3).values
support_levels = data['low'].iloc[peaks_low].nsmallest(3).values

print(f"\n🎯 Current {symbol} Price: ${data['close'].iloc[-1]:,.2f}")
print(f"📊 RSI: {data['RSI'].iloc[-1]:.2f} {'(Overbought)' if data['RSI'].iloc[-1] > 70 else '(Oversold)' if data['RSI'].iloc[-1] < 30 else '(Neutral)'}")
print(f"📈 MACD: {'Bullish' if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1] else 'Bearish'}")
print(f"💪 Volatility (BB Width): {data['BB_width'].iloc[-1]:.4f}")
print(f"\n🔴 Resistance Levels: {', '.join([f'${x:,.0f}' for x in resistance_levels])}")
print(f"🟢 Support Levels: {', '.join([f'${x:,.0f}' for x in support_levels])}")

# Create comprehensive chart
fig = make_subplots(
   rows=5, cols=1,
   shared_xaxes=True,
   vertical_spacing=0.03,
   row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
   subplot_titles=(f'{symbol} Price Action with Bollinger Bands', 'Volume', 'RSI', 'MACD', 'ATR (Volatility)')
)

# Price + Bollinger Bands + Support/Resistance
fig.add_trace(go.Candlestick(
   x=data.index,
   open=data['open'],
   high=data['high'],
   low=data['low'],
   close=data['close'],
   name=symbol,
   increasing_line_color='#00ff41',
   decreasing_line_color='#ff4444'
), row=1, col=1)

fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='cyan', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='yellow', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], name='SMA 200', line=dict(color='red', width=2)), row=1, col=1)

fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], name='BB Upper', line=dict(color='gray', dash='dash', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], name='BB Lower', line=dict(color='gray', dash='dash', width=1), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)

# Add support/resistance lines
for level in resistance_levels:
   fig.add_hline(y=level, line_dash="dot", line_color="red", opacity=0.5, row=1, col=1)
for level in support_levels:
   fig.add_hline(y=level, line_dash="dot", line_color="green", opacity=0.5, row=1, col=1)

# Volume
fig.add_trace(go.Bar(x=data.index, y=data['volume'], name='Volume', marker_color='rgba(255,165,0,0.5)'), row=2, col=1)

# RSI
fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='orange', width=2)), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

# MACD
fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue', width=1.5)), row=4, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['MACD_signal'], name='Signal', line=dict(color='red', width=1.5)), row=4, col=1)
fig.add_trace(go.Bar(x=data.index, y=data['MACD_hist'], name='Histogram', marker_color='gray'), row=4, col=1)

# ATR (Volatility)
fig.add_trace(go.Scatter(x=data.index, y=data['ATR'], name='ATR', line=dict(color='purple', width=2), fill='tozeroy'), row=5, col=1)

fig.update_layout(
   title=f'{symbol} Complete Technical Analysis Dashboard',
   template='plotly_dark',
   height=1400,
   showlegend=True,
   xaxis_rangeslider_visible=False
)

fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)
fig.update_yaxes(title_text="RSI", row=3, col=1)
fig.update_yaxes(title_text="MACD", row=4, col=1)
fig.update_yaxes(title_text="ATR", row=5, col=1)

fig.show()


