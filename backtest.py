"""Walk-forward backtest: replay historical data, generate Kronos signals, simulate trades.

Usage:
    # Demo backtest (synthetic data, no network)
    python backtest.py --demo

    # Backtest on a CSV
    python backtest.py --csv data.csv

    # Control parameters
    python backtest.py --demo --lookback 400 --forecast 24 --step 24 --capital 10000
"""
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, ".")
from model import Kronos, KronosTokenizer, KronosPredictor


def generate_demo_data(n_bars=1200):
    """Generate synthetic BTC-like OHLCV data with trends and mean reversion."""
    np.random.seed(123)
    timestamps = pd.date_range(end=datetime.now(), periods=n_bars, freq="1h")

    close = np.zeros(n_bars)
    close[0] = 60000
    for i in range(1, n_bars):
        trend = np.sin(i / 100) * 30
        noise = np.random.randn() * 120
        mean_rev = (60000 - close[i - 1]) * 0.002
        close[i] = close[i - 1] + trend + noise + mean_rev

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


def load_csv(path):
    """Load OHLCV data from CSV."""
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
    return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)


def infer_freq(df):
    """Infer candle frequency from timestamps."""
    if len(df) < 2:
        return "1h"
    delta = df["timestamps"].iloc[1] - df["timestamps"].iloc[0]
    seconds = int(delta.total_seconds())
    freq_map = {60: "1min", 300: "5min", 900: "15min", 3600: "1h", 14400: "4h", 86400: "1D"}
    return freq_map.get(seconds, f"{seconds}s")


def get_signal(current_price, pred_df, threshold=0.5):
    """Return signal and predicted price change %."""
    avg_pred = pred_df["close"].mean()
    pct = (avg_pred - current_price) / current_price * 100
    if pct > threshold:
        return "BUY", pct
    elif pct < -threshold:
        return "SELL", pct
    return "HOLD", pct


def run_backtest(df, predictor, lookback, forecast, step, capital, threshold, samples, verbose):
    """Walk forward through data, predict, trade, and track results."""
    freq = infer_freq(df)
    n = len(df)
    max_start = n - forecast

    trades = []
    equity_curve = []
    position = None  # None = flat, dict = in a trade
    cash = capital
    total_steps = 0

    i = lookback
    while i < max_start:
        total_steps += 1
        window = df.iloc[i - lookback:i].reset_index(drop=True)
        actual_future = df.iloc[i:i + forecast].reset_index(drop=True)

        x_df = window[["open", "high", "low", "close", "volume", "amount"]]
        x_timestamp = window["timestamps"]
        last_ts = window["timestamps"].iloc[-1]
        y_timestamp = pd.Series(pd.date_range(
            start=last_ts + pd.Timedelta(freq),
            periods=forecast,
            freq=freq,
        ))

        current_price = window["close"].iloc[-1]

        try:
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=forecast,
                T=1.0,
                top_p=0.9,
                sample_count=samples,
                verbose=False,
            )
            signal, pct_pred = get_signal(current_price, pred_df, threshold)
        except Exception as e:
            if verbose:
                print(f"  Step {total_steps}: prediction failed ({e}), skipping")
            i += step
            continue

        actual_future_close = actual_future["close"].iloc[-1] if len(actual_future) == forecast else None

        if signal == "BUY" and position is None and actual_future_close is not None:
            position = {
                "entry_price": current_price,
                "entry_step": total_steps,
                "entry_idx": i,
            }
            if verbose:
                print(f"  Step {total_steps} | idx {i} | BUY  @ ${current_price:,.2f} | pred {pct_pred:+.2f}%")

        elif signal == "SELL" and position is not None:
            exit_price = current_price
            pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
            pnl_dollar = cash * (pnl_pct / 100)
            cash += pnl_dollar
            trades.append({
                "entry_step": position["entry_step"],
                "exit_step": total_steps,
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "pnl_dollar": pnl_dollar,
                "hold_bars": i - position["entry_idx"],
            })
            if verbose:
                print(f"  Step {total_steps} | idx {i} | SELL @ ${exit_price:,.2f} | P&L: {pnl_pct:+.2f}% (${pnl_dollar:+,.2f})")
            position = None

        elif verbose and signal != "HOLD":
            print(f"  Step {total_steps} | idx {i} | {signal} @ ${current_price:,.2f} | pred {pct_pred:+.2f}% (no action)")

        equity_curve.append({"step": total_steps, "idx": i, "cash": cash, "price": current_price})
        i += step

    # Close any open position at the end
    if position is not None:
        exit_price = df["close"].iloc[min(i, n - 1)]
        pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"] * 100
        pnl_dollar = cash * (pnl_pct / 100)
        cash += pnl_dollar
        trades.append({
            "entry_step": position["entry_step"],
            "exit_step": total_steps,
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "pnl_dollar": pnl_dollar,
            "hold_bars": i - position["entry_idx"],
        })
        if verbose:
            print(f"  [END]  Closed open position @ ${exit_price:,.2f} | P&L: {pnl_pct:+.2f}%")

    return trades, equity_curve, cash, total_steps


def compute_metrics(trades, equity_curve, initial_capital, final_capital, df, lookback):
    """Compute and return performance metrics."""
    # Buy & hold benchmark
    start_price = df["close"].iloc[lookback]
    end_price = df["close"].iloc[-1]
    bnh_return = (end_price - start_price) / start_price * 100

    total_return = (final_capital - initial_capital) / initial_capital * 100

    if not trades:
        return {
            "total_return": total_return,
            "buy_hold_return": bnh_return,
            "num_trades": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "max_drawdown": 0,
            "profit_factor": 0,
            "avg_hold_bars": 0,
        }

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]

    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    avg_hold = np.mean([t["hold_bars"] for t in trades])

    gross_profit = sum(t["pnl_dollar"] for t in wins)
    gross_loss = abs(sum(t["pnl_dollar"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown from equity curve
    peak = initial_capital
    max_dd = 0
    for point in equity_curve:
        if point["cash"] > peak:
            peak = point["cash"]
        dd = (peak - point["cash"]) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        "total_return": total_return,
        "buy_hold_return": bnh_return,
        "num_trades": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "avg_hold_bars": avg_hold,
    }


def print_results(metrics, capital, final_capital, total_steps):
    """Print formatted backtest results."""
    m = metrics
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Steps evaluated:    {total_steps}")
    print(f"  Trades executed:    {m['num_trades']}")
    print(f"  Starting Capital:   ${capital:,.2f}")
    print(f"  Final Capital:      ${final_capital:,.2f}")
    print("-" * 60)
    print(f"  Strategy Return:    {m['total_return']:+.2f}%")
    print(f"  Buy & Hold Return:  {m['buy_hold_return']:+.2f}%")
    alpha = m['total_return'] - m['buy_hold_return']
    print(f"  Alpha (vs B&H):     {alpha:+.2f}%")
    print("-" * 60)
    if m['num_trades'] > 0:
        print(f"  Win Rate:           {m['win_rate']:.1f}%")
        print(f"  Avg Win:            {m['avg_win']:+.2f}%")
        print(f"  Avg Loss:           {m['avg_loss']:+.2f}%")
        print(f"  Profit Factor:      {m['profit_factor']:.2f}")
        print(f"  Max Drawdown:       {m['max_drawdown']:.2f}%")
        print(f"  Avg Hold (bars):    {m['avg_hold_bars']:.0f}")
    print("=" * 60)

    if m['num_trades'] == 0:
        print("  No trades were triggered. Try lowering --threshold.")
    elif m['total_return'] > m['buy_hold_return']:
        print("  Strategy OUTPERFORMED buy & hold.")
    else:
        print("  Strategy UNDERPERFORMED buy & hold.")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Kronos Backtester")
    parser.add_argument("--csv", help="CSV file with OHLCV data")
    parser.add_argument("--demo", action="store_true", help="Use synthetic data")
    parser.add_argument("--lookback", type=int, default=400, help="Bars of context per prediction")
    parser.add_argument("--forecast", type=int, default=24, help="Bars ahead to forecast")
    parser.add_argument("--step", type=int, default=24, help="Bars to advance between predictions")
    parser.add_argument("--capital", type=float, default=10000, help="Starting capital ($)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Signal threshold (%%)")
    parser.add_argument("--samples", type=int, default=1, help="Forecast samples per step")
    parser.add_argument("--model", default="NeoQuasar/Kronos-small", help="HuggingFace model")
    parser.add_argument("--bars", type=int, default=800, help="Total bars of demo data to generate")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print each trade")
    args = parser.parse_args()

    if not args.csv and not args.demo:
        print("Specify --csv <file> or --demo")
        sys.exit(1)

    print("Loading Kronos model...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    print("Model loaded.\n")

    if args.demo:
        print(f"Generating {args.bars} bars of synthetic data...")
        df = generate_demo_data(args.bars)
    else:
        print(f"Loading {args.csv}...")
        df = load_csv(args.csv)

    n = len(df)
    test_bars = n - args.lookback
    print(f"Total bars: {n}")
    print(f"Lookback: {args.lookback}, Forecast: {args.forecast}, Step: {args.step}")
    print(f"Test window: {test_bars} bars (~{test_bars // args.step} predictions)")
    print(f"Threshold: {args.threshold}%, Samples: {args.samples}")
    print()

    trades, equity_curve, final_capital, total_steps = run_backtest(
        df, predictor, args.lookback, args.forecast, args.step,
        args.capital, args.threshold, args.samples, args.verbose,
    )

    metrics = compute_metrics(trades, equity_curve, args.capital, final_capital, df, args.lookback)
    print_results(metrics, args.capital, final_capital, total_steps)

    if trades:
        print("Trade log:")
        print(f"  {'#':>3}  {'Entry':>12}  {'Exit':>12}  {'P&L':>8}  {'Bars':>5}")
        print(f"  {'---':>3}  {'----------':>12}  {'----------':>12}  {'------':>8}  {'----':>5}")
        for i, t in enumerate(trades, 1):
            print(f"  {i:3d}  ${t['entry_price']:>10,.2f}  ${t['exit_price']:>10,.2f}  {t['pnl_pct']:>+7.2f}%  {t['hold_bars']:>5d}")


if __name__ == "__main__":
    main()
