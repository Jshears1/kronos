"""Pull market data, run Kronos forecast, and generate trading signals.

Usage:
    # Live data (requires ccxt: pip install ccxt)
    python signals.py --symbol BTC/USDT --timeframe 1h --forecast 24

    # From a CSV file
    python signals.py --csv data.csv --forecast 24

    # Demo mode (synthetic data, no network needed)
    python signals.py --demo
"""
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, ".")
from model import Kronos, KronosTokenizer, KronosPredictor


def fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=500):
    """Fetch OHLCV data via ccxt with automatic exchange fallback for US users."""
    try:
        import ccxt
    except ImportError:
        print("ccxt not installed. Run: pip install ccxt")
        sys.exit(1)

    for eid in ["kraken", "kucoin", "binanceus", "binance", "coinbasepro"]:
        try:
            ex_class = getattr(ccxt, eid, None)
            if ex_class is None:
                continue
            exchange = ex_class({"enableRateLimit": True})
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if bars:
                print(f"Using exchange: {eid}")
                break
        except Exception:
            continue
    else:
        print("All exchanges failed. Try BTC/USD for Kraken, or use --csv / --demo.")
        sys.exit(1)

    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamps"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["amount"] = df["volume"] * df["close"]
    df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
    return df


def load_csv(path):
    """Load OHLCV data from a CSV file.

    Expected columns: timestamps (or date/datetime), open, high, low, close, volume.
    'amount' is computed as volume * close if missing.
    """
    df = pd.read_csv(path)

    ts_col = None
    for col in ["timestamps", "timestamp", "date", "datetime", "time"]:
        if col in df.columns:
            ts_col = col
            break
    if ts_col is None:
        print("CSV must have a timestamps/date/datetime column.")
        sys.exit(1)

    df["timestamps"] = pd.to_datetime(df[ts_col])
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            print(f"CSV missing required column: {col}")
            sys.exit(1)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "amount" not in df.columns:
        df["amount"] = df["volume"] * df["close"]

    df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
    return df


def generate_demo_data(n_bars=500):
    """Generate synthetic BTC-like OHLCV data for demo/testing."""
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


def generate_signal(current_price, pred_df):
    """Analyze forecast and return a trading signal."""
    pred_closes = pred_df["close"].values
    avg_pred = np.mean(pred_closes)
    final_pred = pred_closes[-1]

    pct_change_avg = (avg_pred - current_price) / current_price * 100
    pct_change_final = (final_pred - current_price) / current_price * 100

    pred_high = pred_df["high"].max()
    pred_low = pred_df["low"].min()
    upside = (pred_high - current_price) / current_price * 100
    downside = (current_price - pred_low) / current_price * 100

    if pct_change_avg > 0.5:
        signal = "BUY"
    elif pct_change_avg < -0.5:
        signal = "SELL"
    else:
        signal = "HOLD"

    strength = abs(pct_change_avg)
    if strength > 2.0:
        confidence = "STRONG"
    elif strength > 1.0:
        confidence = "MODERATE"
    else:
        confidence = "WEAK"

    return {
        "signal": signal,
        "confidence": confidence,
        "current_price": current_price,
        "avg_predicted": avg_pred,
        "final_predicted": final_pred,
        "pct_change_avg": pct_change_avg,
        "pct_change_final": pct_change_final,
        "predicted_high": pred_high,
        "predicted_low": pred_low,
        "upside_pct": upside,
        "downside_pct": downside,
    }


def print_signal(result, symbol, timeframe, pred_len):
    """Print a formatted trading signal summary."""
    print("\n" + "=" * 60)
    print(f"  KRONOS SIGNAL: {symbol} ({timeframe} candles)")
    print("=" * 60)
    print(f"  Signal:          {result['signal']} ({result['confidence']})")
    print(f"  Current Price:   ${result['current_price']:,.2f}")
    print(f"  Avg Forecast:    ${result['avg_predicted']:,.2f} ({result['pct_change_avg']:+.2f}%)")
    print(f"  Final Forecast:  ${result['final_predicted']:,.2f} ({result['pct_change_final']:+.2f}%)")
    print(f"  Predicted High:  ${result['predicted_high']:,.2f} ({result['upside_pct']:+.2f}% upside)")
    print(f"  Predicted Low:   ${result['predicted_low']:,.2f} ({result['downside_pct']:+.2f}% downside)")
    print(f"  Forecast Window: {pred_len} candles ahead")
    print("=" * 60)
    print("  WARNING: This is a model prediction, not financial advice.")
    print("=" * 60 + "\n")


def infer_freq(df):
    """Infer candle frequency from timestamps."""
    if len(df) < 2:
        return "1h"
    delta = df["timestamps"].iloc[1] - df["timestamps"].iloc[0]
    seconds = int(delta.total_seconds())
    freq_map = {60: "1min", 300: "5min", 900: "15min", 3600: "1h", 14400: "4h", 86400: "1D"}
    return freq_map.get(seconds, f"{seconds}s")


def main():
    parser = argparse.ArgumentParser(description="Kronos Trading Signals")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair (default: BTC/USDT)")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe (default: 1h)")
    parser.add_argument("--csv", help="Path to CSV file with OHLCV data")
    parser.add_argument("--demo", action="store_true", help="Use synthetic data (no network needed)")
    parser.add_argument("--lookback", type=int, default=400, help="Bars of history to feed model")
    parser.add_argument("--forecast", type=int, default=24, help="Bars ahead to forecast")
    parser.add_argument("--samples", type=int, default=3, help="Forecast samples to average (more = smoother)")
    parser.add_argument("--model", default="NeoQuasar/Kronos-small", help="HuggingFace model name")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    args = parser.parse_args()

    print("Loading Kronos model...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    print("Model loaded.\n")

    if args.demo:
        print("Generating synthetic BTC/USDT data (demo mode)...")
        df = generate_demo_data(args.lookback + 50)
        args.symbol = "BTC/USDT (demo)"
        args.timeframe = "1h"
    elif args.csv:
        print(f"Loading data from {args.csv}...")
        df = load_csv(args.csv)
        args.timeframe = infer_freq(df)
    else:
        print(f"Fetching {args.symbol} {args.timeframe} data...")
        try:
            df = fetch_ohlcv(args.symbol, args.timeframe, limit=args.lookback + 50)
        except Exception as e:
            print(f"Failed to fetch live data: {e}")
            print("\nOptions:")
            print("  1. pip install ccxt  (for live exchange data)")
            print("  2. python signals.py --csv yourdata.csv")
            print("  3. python signals.py --demo  (synthetic data)")
            sys.exit(1)

    print(f"Got {len(df)} candles. Latest: {df['timestamps'].iloc[-1]}")

    n = len(df)
    lookback = min(args.lookback, n)
    x_df = df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    x_timestamp = df.iloc[-lookback:]["timestamps"].reset_index(drop=True)

    last_ts = df["timestamps"].iloc[-1]
    freq = infer_freq(df)
    y_timestamp = pd.Series(pd.date_range(
        start=last_ts + pd.Timedelta(freq),
        periods=args.forecast,
        freq=freq,
    ))

    print(f"\nForecasting {args.forecast} candles ahead (samples={args.samples})...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=args.forecast,
        T=args.temperature,
        top_p=0.9,
        sample_count=args.samples,
        verbose=True,
    )

    current_price = df["close"].iloc[-1]
    result = generate_signal(current_price, pred_df)
    print_signal(result, args.symbol, args.timeframe, args.forecast)

    print("Forecast detail:")
    print(pred_df[["open", "high", "low", "close"]].to_string())


if __name__ == "__main__":
    main()
