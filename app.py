"""Kronos Trading Dashboard -- local Streamlit app.

Run: streamlit run app.py
"""
import sys
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

sys.path.insert(0, ".")
from model import Kronos, KronosTokenizer, KronosPredictor


@st.cache_resource
def load_model(model_name):
    """Load and cache the Kronos model (persists across reruns)."""
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(model_name)
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    return predictor


def fetch_live_data(symbol, timeframe, limit):
    """Fetch OHLCV from Binance via ccxt."""
    import ccxt
    exchange = ccxt.binance({"enableRateLimit": True})
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamps"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["amount"] = df["volume"] * df["close"]
    return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]


def generate_demo_data(n_bars=800):
    """Synthetic BTC-like data for testing."""
    np.random.seed(42)
    timestamps = pd.date_range(end=datetime.now(), periods=n_bars, freq="1h")
    close = 60000 + np.cumsum(np.random.randn(n_bars) * 100)
    high = close + np.abs(np.random.randn(n_bars) * 200)
    low = close - np.abs(np.random.randn(n_bars) * 200)
    open_ = close + np.random.randn(n_bars) * 50
    volume = np.abs(np.random.randn(n_bars) * 1000 + 5000)
    amount = volume * close
    return pd.DataFrame({
        "timestamps": timestamps,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "amount": amount,
    })


def infer_freq(df):
    if len(df) < 2:
        return "1h"
    delta = df["timestamps"].iloc[1] - df["timestamps"].iloc[0]
    seconds = int(delta.total_seconds())
    return {60: "1min", 300: "5min", 900: "15min", 3600: "1h", 14400: "4h", 86400: "1D"}.get(seconds, f"{seconds}s")


def run_forecast(predictor, df, lookback, forecast, samples, temperature):
    """Run Kronos prediction and return forecast DataFrame."""
    n = len(df)
    lookback = min(lookback, n)
    window = df.iloc[-lookback:].reset_index(drop=True)

    x_df = window[["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = window["timestamps"]
    last_ts = window["timestamps"].iloc[-1]
    freq = infer_freq(df)

    y_timestamp = pd.Series(pd.date_range(
        start=last_ts + pd.Timedelta(freq),
        periods=forecast,
        freq=freq,
    ))

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=forecast,
        T=temperature,
        top_p=0.9,
        sample_count=samples,
        verbose=False,
    )
    return pred_df


def get_signal(current_price, pred_df, threshold=0.5):
    avg_pred = pred_df["close"].mean()
    pct = (avg_pred - current_price) / current_price * 100
    if pct > threshold:
        return "BUY", pct
    elif pct < -threshold:
        return "SELL", pct
    return "HOLD", pct


def run_backtest(predictor, df, lookback, forecast, step, capital, threshold, samples):
    """Walk-forward backtest returning trades and equity curve."""
    freq = infer_freq(df)
    n = len(df)
    max_start = n - forecast

    trades = []
    equity = []
    position = None
    cash = capital

    i = lookback
    progress = st.progress(0)
    total_steps = max(1, (max_start - lookback) // step)
    step_count = 0

    while i < max_start:
        step_count += 1
        progress.progress(min(step_count / total_steps, 1.0))

        window = df.iloc[i - lookback:i].reset_index(drop=True)
        x_df = window[["open", "high", "low", "close", "volume", "amount"]]
        x_timestamp = window["timestamps"]
        last_ts = window["timestamps"].iloc[-1]

        y_timestamp = pd.Series(pd.date_range(
            start=last_ts + pd.Timedelta(freq),
            periods=forecast,
            freq=freq,
        ))

        current_price = window["close"].iloc[-1]
        current_time = window["timestamps"].iloc[-1]

        try:
            pred_df = predictor.predict(
                df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                pred_len=forecast, T=1.0, top_p=0.9, sample_count=samples, verbose=False,
            )
            signal, pct_pred = get_signal(current_price, pred_df, threshold)
        except Exception:
            i += step
            continue

        if signal == "BUY" and position is None:
            position = {"entry_price": current_price, "entry_idx": i, "entry_time": current_time}
        elif signal == "SELL" and position is not None:
            pnl_pct = (current_price - position["entry_price"]) / position["entry_price"] * 100
            pnl_dollar = cash * (pnl_pct / 100)
            cash += pnl_dollar
            trades.append({
                "entry_time": position["entry_time"],
                "exit_time": current_time,
                "entry_price": position["entry_price"],
                "exit_price": current_price,
                "pnl_pct": pnl_pct,
                "pnl_dollar": pnl_dollar,
            })
            position = None

        equity.append({"time": current_time, "equity": cash, "price": current_price})
        i += step

    if position is not None:
        exit_price = df["close"].iloc[-1]
        pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
        pnl_dollar = cash * (pnl_pct / 100)
        cash += pnl_dollar
        trades.append({
            "entry_time": position["entry_time"],
            "exit_time": df["timestamps"].iloc[-1],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "pnl_dollar": pnl_dollar,
        })

    progress.empty()
    return trades, pd.DataFrame(equity), cash


# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Kronos Trading", page_icon="📈", layout="wide")
st.title("Kronos Trading Dashboard")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox("Model", [
        "NeoQuasar/Kronos-small",
        "NeoQuasar/Kronos-base",
        "NeoQuasar/Kronos-mini",
    ])

    data_source = st.radio("Data Source", ["Live (ccxt)", "CSV Upload", "Demo (synthetic)"])

    symbol = "BTC/USDT"
    timeframe = "1h"
    if data_source == "Live (ccxt)":
        symbol = st.text_input("Symbol", "BTC/USDT")
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)

    uploaded_file = None
    if data_source == "CSV Upload":
        uploaded_file = st.file_uploader("Upload OHLCV CSV", type=["csv"])

    st.divider()
    lookback = st.slider("Lookback (bars)", 100, 500, 400)
    forecast = st.slider("Forecast (bars)", 6, 96, 24)
    samples = st.slider("Samples (smoothness)", 1, 10, 3)
    temperature = st.slider("Temperature", 0.5, 2.0, 1.0, 0.1)
    threshold = st.slider("Signal threshold (%)", 0.1, 2.0, 0.5, 0.1)

# ── Load model ───────────────────────────────────────────────
with st.spinner("Loading Kronos model..."):
    predictor = load_model(model_name)

# ── Load data ────────────────────────────────────────────────
df = None
if data_source == "Demo (synthetic)":
    df = generate_demo_data(800)
    st.info("Using synthetic BTC/USDT data (demo mode)")
elif data_source == "CSV Upload" and uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    for col in ["timestamps", "timestamp", "date", "datetime"]:
        if col in df.columns:
            df["timestamps"] = pd.to_datetime(df[col])
            break
    if "amount" not in df.columns:
        df["amount"] = df.get("volume", 0) * df["close"]
    df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
    st.success(f"Loaded {len(df)} rows from CSV")
elif data_source == "Live (ccxt)":
    try:
        with st.spinner(f"Fetching {symbol} {timeframe} data..."):
            df = fetch_live_data(symbol, timeframe, lookback + 50)
        st.success(f"Fetched {len(df)} candles")
    except Exception as e:
        st.error(f"Failed to fetch data: {e}\n\nMake sure ccxt is installed: `pip install ccxt`")

if df is None:
    st.warning("Select a data source to get started.")
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────
tab_signal, tab_backtest, tab_data = st.tabs(["Signal", "Backtest", "Raw Data"])

# ── Signal tab ───────────────────────────────────────────────
with tab_signal:
    if st.button("Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Running Kronos forecast..."):
            pred_df = run_forecast(predictor, df, lookback, forecast, samples, temperature)

        current_price = df["close"].iloc[-1]
        signal, pct = get_signal(current_price, pred_df, threshold)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            color = {"BUY": "green", "SELL": "red", "HOLD": "orange"}[signal]
            st.markdown(f"### :{color}[{signal}]")
        with col2:
            st.metric("Current Price", f"${current_price:,.2f}")
        with col3:
            avg_pred = pred_df["close"].mean()
            st.metric("Avg Forecast", f"${avg_pred:,.2f}", f"{pct:+.2f}%")
        with col4:
            final_pred = pred_df["close"].iloc[-1]
            final_pct = (final_pred - current_price) / current_price * 100
            st.metric("Final Forecast", f"${final_pred:,.2f}", f"{final_pct:+.2f}%")

        st.divider()

        # Price chart with forecast
        history_tail = df.iloc[-min(100, len(df)):][["timestamps", "close"]].copy()
        history_tail = history_tail.rename(columns={"close": "Actual", "timestamps": "Time"}).set_index("Time")

        forecast_series = pred_df[["close"]].copy()
        forecast_series.index.name = "Time"
        forecast_series = forecast_series.rename(columns={"close": "Forecast"})

        chart_df = pd.concat([history_tail, forecast_series], axis=0)
        st.line_chart(chart_df, use_container_width=True)

        # Forecast detail
        with st.expander("Forecast detail"):
            st.dataframe(pred_df[["open", "high", "low", "close", "volume"]], use_container_width=True)

        # Range info
        col_a, col_b = st.columns(2)
        with col_a:
            pred_high = pred_df["high"].max()
            up_pct = (pred_high - current_price) / current_price * 100
            st.metric("Predicted High", f"${pred_high:,.2f}", f"{up_pct:+.2f}% upside")
        with col_b:
            pred_low = pred_df["low"].min()
            down_pct = (current_price - pred_low) / current_price * 100
            st.metric("Predicted Low", f"${pred_low:,.2f}", f"-{down_pct:.2f}% downside")

    else:
        st.write("Click **Generate Forecast** to run the Kronos model on the loaded data.")
        st.line_chart(df.set_index("timestamps")["close"].tail(200), use_container_width=True)


# ── Backtest tab ─────────────────────────────────────────────
with tab_backtest:
    col_bt1, col_bt2 = st.columns(2)
    with col_bt1:
        bt_step = st.number_input("Step size (bars)", min_value=6, max_value=96, value=24)
    with col_bt2:
        bt_capital = st.number_input("Starting capital ($)", min_value=100, max_value=1000000, value=10000)

    if st.button("Run Backtest", type="primary", use_container_width=True):
        with st.spinner("Running walk-forward backtest... (this takes a while)"):
            trades, equity_df, final_capital = run_backtest(
                predictor, df, lookback, forecast, bt_step, bt_capital, threshold, samples=1,
            )

        # Metrics
        total_return = (final_capital - bt_capital) / bt_capital * 100
        start_price = df["close"].iloc[lookback]
        end_price = df["close"].iloc[-1]
        bnh_return = (end_price - start_price) / start_price * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Strategy Return", f"{total_return:+.2f}%")
        with col2:
            st.metric("Buy & Hold", f"{bnh_return:+.2f}%")
        with col3:
            st.metric("Alpha", f"{total_return - bnh_return:+.2f}%")
        with col4:
            st.metric("Trades", len(trades))

        if trades:
            wins = [t for t in trades if t["pnl_pct"] > 0]
            win_rate = len(wins) / len(trades) * 100

            col5, col6, col7 = st.columns(3)
            with col5:
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col6:
                avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
                st.metric("Avg Win", f"{avg_win:+.2f}%")
            with col7:
                losses = [t for t in trades if t["pnl_pct"] <= 0]
                avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
                st.metric("Avg Loss", f"{avg_loss:+.2f}%")

            st.divider()

            if not equity_df.empty:
                st.subheader("Equity Curve")
                st.line_chart(equity_df.set_index("time")["equity"], use_container_width=True)

            st.subheader("Trade Log")
            trade_df = pd.DataFrame(trades)
            trade_df["pnl_pct"] = trade_df["pnl_pct"].map(lambda x: f"{x:+.2f}%")
            trade_df["pnl_dollar"] = trade_df["pnl_dollar"].map(lambda x: f"${x:+,.2f}")
            trade_df["entry_price"] = trade_df["entry_price"].map(lambda x: f"${x:,.2f}")
            trade_df["exit_price"] = trade_df["exit_price"].map(lambda x: f"${x:,.2f}")
            st.dataframe(trade_df, use_container_width=True)
        else:
            st.warning("No trades triggered. Try lowering the signal threshold.")
    else:
        st.write("Configure parameters and click **Run Backtest** to simulate trading.")


# ── Raw Data tab ─────────────────────────────────────────────
with tab_data:
    st.subheader(f"Loaded Data ({len(df)} rows)")
    st.dataframe(df.tail(100), use_container_width=True)

    st.download_button(
        "Download as CSV",
        df.to_csv(index=False),
        file_name="kronos_data.csv",
        mime="text/csv",
    )
