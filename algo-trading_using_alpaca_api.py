import alpaca_trade_api as tradeapi        
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from datetime import datetime, timedelta

# 1. AUTHENTICATION (Use Paper Trading Keys)
API_KEY = 'PKKULZ27OKPJRDXSAZ7RKL5FLP'
SECRET_KEY = 'BPR1n5npfNWTWbMy9BDaY4kHfDEY2wWoyACDVFirxCsW'
BASE_URL = 'https://paper-api.alpaca.markets' # for live trading session we will use 
# https://api.alpaca.markets

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')



def get_live_data(symbol):
    # 1. Calculate a start date (e.g., 400 days ago to be very safe)
    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
    
    # 2. Fetch bars WITH the start parameter
    # Note: '1Day' timeframe needs a 'start' string in YYYY-MM-DD format
    bars = api.get_bars(symbol, '1Day', start=start_date, limit=400).df
    
    print(f"Data rows retrieved: {len(bars)}") 
    print(bars.head())
    return bars

def execute_trade(symbol, signal):
    # 1. Check for OPEN ORDERS first (Don't buy if an order is already pending)
    open_orders = api.list_orders(status='open', symbols=[symbol])
    if len(open_orders) > 0:
        print(f"Order for {symbol} already pending. Skipping...")
        return

    # 2. Check current position
    try:
        api.get_position(symbol)
        holding = True
    except:
        holding = False

    # 3. Calculate Position Sizing
    if signal == 1 and not holding:
        # Get account info to see how much cash we have
        account = api.get_account()
        cash = float(account.cash)
        
        # Get the latest price to calculate quantity
        # Let's say we want to risk 10% of our total cash on this trade
        last_price = float(api.get_latest_trade(symbol).price)
        quantity = int((cash * 0.10) / last_price)

        if quantity > 0:
            api.submit_order(
                symbol=symbol, 
                qty=quantity, 
                side='buy', 
                type='market', 
                time_in_force='gtc'
            )
            print(f"BUY ORDER PLACED: {quantity} shares at approx ${last_price}")
        else:
            print("Not enough cash to buy even 1 share.")

    elif signal == 0 and holding:
        # Get the quantity we currently hold to sell it all
        position = api.get_position(symbol)
        api.submit_order(
            symbol=symbol, 
            qty=position.qty, 
            side='sell', 
            type='market', 
            time_in_force='gtc'
        )
        print(f"SELL ORDER PLACED: {position.qty} shares")

# 2. THE LIVE LOOP


while True:
    print("Fetching new data...")
    df = get_live_data("AAPL")
    sma50 = df['close'].rolling(50).mean().iloc[-1]
    sma200 = df['close'].rolling(200).mean().iloc[-1]
    
    print(f"Current SMA50: {sma50:.2f} | SMA200: {sma200:.2f}")
    current_signal = 1 if sma50 > sma200 else 0
    execute_trade("AAPL", current_signal)
    
    print("Going to sleep for 60 seconds...")
    time.sleep(60)  #wait for next 24 hour daily candle