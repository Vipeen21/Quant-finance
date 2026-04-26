import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openbb import obb
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.precision", 4)

ASSET_CONFIG = {
    "stocks": {
        "label": "Stock symbol",
        "default_symbol": "AAPL",
        "placeholder": "AAPL",
        "fetcher": lambda symbol, start_date: obb.equity.price.historical(
            symbol=symbol, provider="yfinance", start_date=start_date
        ).to_df(),
    },
    "crypto": {
        "label": "Crypto symbol",
        "default_symbol": "BTCUSD",
        "placeholder": "BTCUSD",
        "fetcher": lambda symbol, start_date: obb.crypto.price.historical(
            symbol=symbol, provider="yfinance", start_date=start_date
        ).to_df(),
    },
    "forex": {
        "label": "Forex pair",
        "default_symbol": "EURUSD",
        "placeholder": "EURUSD",
        "fetcher": lambda symbol, start_date: obb.currency.price.historical(
            symbol=symbol, provider="yfinance", start_date=start_date
        ).to_df(),
    },
}


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy().sort_index()

    df["SMA_20"] = df["close"].rolling(20).mean()
    df["SMA_50"] = df["close"].rolling(50).mean()
    df["SMA_200"] = df["close"].rolling(200).mean()

    df["BB_middle"] = df["close"].rolling(20).mean()
    df["BB_std"] = df["close"].rolling(20).std()
    df["BB_upper"] = df["BB_middle"] + (df["BB_std"] * 2)
    df["BB_lower"] = df["BB_middle"] - (df["BB_std"] * 2)
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_middle"]

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    previous_close = df["close"].shift(1)
    df["TR"] = np.maximum(
        df["high"] - df["low"],
        np.maximum((df["high"] - previous_close).abs(), (df["low"] - previous_close).abs()),
    )
    df["ATR"] = df["TR"].rolling(14).mean()

    return df


def get_support_resistance_levels(data: pd.DataFrame) -> tuple[list[float], list[float]]:
    high_prominence = data["high"].std()
    low_prominence = data["low"].std()

    peaks_high, _ = find_peaks(data["high"], distance=20, prominence=high_prominence)
    peaks_low, _ = find_peaks(-data["low"], distance=20, prominence=low_prominence)

    resistance_levels = data["high"].iloc[peaks_high].nlargest(3).tolist()
    support_levels = data["low"].iloc[peaks_low].nsmallest(3).tolist()

    return resistance_levels, support_levels


@st.cache_data(show_spinner=False)
def fetch_market_data(asset_class: str, symbol: str, lookback_days: int) -> pd.DataFrame:
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    fetcher = ASSET_CONFIG[asset_class]["fetcher"]
    data = fetcher(symbol, start_date)

    if data.empty:
        raise ValueError("No market data returned for the selected asset.")

    data.index = pd.to_datetime(data.index)
    return data


def build_dashboard_figure(
    data: pd.DataFrame, symbol: str, resistance_levels: list[float], support_levels: list[float]
) -> go.Figure:
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
        subplot_titles=(
            f"{symbol} Price Action with Bollinger Bands",
            "Volume",
            "RSI",
            "MACD",
            "ATR (Volatility)",
        ),
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name=symbol,
            increasing_line_color="#00ff41",
            decreasing_line_color="#ff4444",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=data.index, y=data["SMA_20"], name="SMA 20", line=dict(color="cyan", width=1)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["SMA_50"], name="SMA 50", line=dict(color="yellow", width=1)
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["SMA_200"], name="SMA 200", line=dict(color="red", width=2)),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["BB_upper"],
            name="BB Upper",
            line=dict(color="gray", dash="dash", width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["BB_lower"],
            name="BB Lower",
            line=dict(color="gray", dash="dash", width=1),
            fill="tonexty",
            fillcolor="rgba(128,128,128,0.1)",
        ),
        row=1,
        col=1,
    )

    for level in resistance_levels:
        fig.add_hline(y=level, line_dash="dot", line_color="red", opacity=0.5, row=1, col=1)

    for level in support_levels:
        fig.add_hline(y=level, line_dash="dot", line_color="green", opacity=0.5, row=1, col=1)

    fig.add_trace(
        go.Bar(x=data.index, y=data["volume"], name="Volume", marker_color="rgba(255,165,0,0.5)"),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=data.index, y=data["RSI"], name="RSI", line=dict(color="orange", width=2)),
        row=3,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

    fig.add_trace(
        go.Scatter(x=data.index, y=data["MACD"], name="MACD", line=dict(color="blue", width=1.5)),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data["MACD_signal"], name="Signal", line=dict(color="red", width=1.5)
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=data.index, y=data["MACD_hist"], name="Histogram", marker_color="gray"),
        row=4,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=data.index, y=data["ATR"], name="ATR", line=dict(color="purple", width=2), fill="tozeroy"),
        row=5,
        col=1,
    )

    fig.update_layout(
        title=f"{symbol} Complete Technical Analysis Dashboard",
        template="plotly_dark",
        height=1400,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=30, r=30, t=70, b=30),
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    fig.update_yaxes(title_text="ATR", row=5, col=1)

    return fig


def rsi_label(rsi_value: float) -> str:
    if rsi_value > 70:
        return "Overbought"
    if rsi_value < 30:
        return "Oversold"
    return "Neutral"


def macd_label(data: pd.DataFrame) -> str:
    return "Bullish" if data["MACD"].iloc[-1] > data["MACD_signal"].iloc[-1] else "Bearish"


def main() -> None:
    st.set_page_config(page_title="OpenBB Technical Analysis", layout="wide")
    st.title("Technical Analysis Web App")
    st.caption("Explore stocks, crypto, and forex with OpenBB, Plotly, and live indicator summaries.")

    with st.sidebar:
        st.header("Market Input")
        asset_class = st.selectbox("Asset class", options=list(ASSET_CONFIG.keys()), index=1)
        config = ASSET_CONFIG[asset_class]
        symbol = st.text_input(
            config["label"],
            value=config["default_symbol"],
            placeholder=config["placeholder"],
        ).strip().upper()
        lookback_days = st.slider("Lookback window (days)", min_value=90, max_value=730, value=365, step=30)
        run_analysis = st.button("Run analysis", type="primary", use_container_width=True)

    if not run_analysis:
        st.info("Choose an asset and click Run analysis to build the dashboard.")
        return

    if not symbol:
        st.warning("Enter a valid symbol to continue.")
        return

    try:
        raw_data = fetch_market_data(asset_class, symbol, lookback_days)
        data = add_indicators(raw_data)
        resistance_levels, support_levels = get_support_resistance_levels(data)
        figure = build_dashboard_figure(data, symbol, resistance_levels, support_levels)
    except Exception as exc:
        st.error(f"Unable to load analysis for {symbol}: {exc}")
        return

    latest_close = data["close"].iloc[-1]
    latest_rsi = data["RSI"].iloc[-1]
    latest_bb_width = data["BB_width"].iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${latest_close:,.2f}")
    col2.metric("RSI", f"{latest_rsi:.2f}", rsi_label(latest_rsi))
    col3.metric("MACD Signal", macd_label(data))
    col4.metric("BB Width", f"{latest_bb_width:.4f}")

    support_text = ", ".join(f"${level:,.0f}" for level in support_levels) if support_levels else "Not enough pivots"
    resistance_text = (
        ", ".join(f"${level:,.0f}" for level in resistance_levels) if resistance_levels else "Not enough pivots"
    )

    st.markdown(
        f"""
        **Support levels:** {support_text}  
        **Resistance levels:** {resistance_text}
        """
    )

    st.plotly_chart(figure, use_container_width=True)

    preview_columns = ["open", "high", "low", "close", "volume", "RSI", "MACD", "ATR"]
    st.subheader("Latest data")
    st.dataframe(data[preview_columns].tail(20), use_container_width=True)


if __name__ == "__main__":
    main()
