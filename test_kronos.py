"""Quick test: fetch BTC/USDT data and run a Kronos forecast."""
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, ".")
from model import Kronos, KronosTokenizer, KronosPredictor


def generate_sample_btc_data(n_bars=500):
    """Generate synthetic BTC-like OHLCV data for testing when no API is available."""
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
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": amount,
    })


def main():
    print("Loading Kronos model and tokenizer from HuggingFace...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    print("Model loaded successfully.\n")

    print("Generating sample BTC/USDT hourly data...")
    df = generate_sample_btc_data(500)
    print(f"Data shape: {df.shape}")
    print(df.tail(3))
    print()

    lookback = 400
    pred_len = 24  # forecast next 24 hours

    x_df = df.loc[:lookback - 1, ["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df.loc[:lookback - 1, "timestamps"]
    y_timestamp = pd.date_range(
        start=df.loc[lookback - 1, "timestamps"] + timedelta(hours=1),
        periods=pred_len,
        freq="1h",
    )

    print(f"Forecasting next {pred_len} hours using {lookback} bars of context...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )

    print("\nForecasted OHLCV:")
    print(pred_df.to_string())
    print("\nKronos is working.")


if __name__ == "__main__":
    main()
