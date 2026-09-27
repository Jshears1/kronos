"""Rigorous out-of-sample validation for Kronos trading signals.

Pulls REAL historical data, splits into in-sample/out-of-sample periods,
and runs walk-forward backtest on data the model has never seen during
parameter tuning.

Usage:
    # Pull real BTC data and validate (requires ccxt)
    python validate.py

    # Validate on a specific pair and timeframe
    python validate.py --symbol ETH/USDT --timeframe 4h

    # Use existing CSV (must have: timestamps, open, high, low, close, volume)
    python validate.py --csv real_btc_data.csv

    # Control test split
    python validate.py --test-pct 0.3

    # Save fetched data for reproducibility
    python validate.py --save-data

    # Sweep thresholds to check robustness
    python validate.py --sweep
"""
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, ".")
from model import Kronos, KronosTokenizer, KronosPredictor


def fetch_real_data(symbol="BTC/USDT", timeframe="1h", total_bars=2000, exchange_id=None):
    """Fetch real OHLCV data with automatic exchange fallback for US users."""
    try:
        import ccxt
    except ImportError:
        print("ccxt required: pip install ccxt")
        sys.exit(1)

    # Exchanges that work in the US, in order of preference
    if exchange_id:
        exchange_list = [exchange_id]
    else:
        exchange_list = ["kraken", "kucoin", "binanceus", "binance", "coinbasepro"]

    exchange = None
    for eid in exchange_list:
        try:
            ex_class = getattr(ccxt, eid, None)
            if ex_class is None:
                continue
            ex = ex_class({"enableRateLimit": True})
            # Quick test fetch
            test = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=5)
            if test:
                exchange = ex
                print(f"Using exchange: {eid}")
                break
        except Exception as e:
            print(f"  {eid}: unavailable ({type(e).__name__}: {str(e)[:80]})")
            continue

    if exchange is None:
        print("\nAll exchanges failed. Options:")
        print("  1. Specify an exchange: --exchange kraken")
        print("  2. Some pairs differ by exchange (try BTC/USD instead of BTC/USDT for Kraken)")
        print("  3. Use a CSV: --csv yourdata.csv")
        sys.exit(1)

    # Map timeframe to milliseconds for backward pagination
    tf_ms = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000,
        "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
    }
    candle_ms = tf_ms.get(timeframe, 3_600_000)
    batch_size = 720

    print(f"Fetching {total_bars} bars of {symbol} {timeframe} data...")

    # First fetch: get latest bars (no `since`)
    all_bars = []
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=batch_size)
        if bars:
            all_bars = bars
            print(f"  Fetched {len(all_bars)}/{total_bars} bars...")
    except Exception as e:
        print(f"Fetch error: {e}")

    # Paginate backwards to get older data
    while len(all_bars) < total_bars and all_bars:
        earliest_ts = all_bars[0][0]
        since = earliest_ts - batch_size * candle_ms
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=batch_size)
        except Exception as e:
            print(f"Fetch error: {e}")
            break
        if not bars:
            break
        # Keep only bars older than what we already have
        new_bars = [b for b in bars if b[0] < earliest_ts]
        if not new_bars:
            break
        all_bars = new_bars + all_bars
        print(f"  Fetched {len(all_bars)}/{total_bars} bars...")

    if not all_bars:
        print("No data fetched. Check network and symbol.")
        sys.exit(1)

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamps"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["amount"] = df["volume"] * df["close"]
    df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
    df = df.drop_duplicates(subset="timestamps").sort_values("timestamps").reset_index(drop=True)

    # Trim to requested size (keep latest)
    if len(df) > total_bars:
        df = df.iloc[-total_bars:].reset_index(drop=True)

    print(f"Got {len(df)} unique bars from {df['timestamps'].iloc[0]} to {df['timestamps'].iloc[-1]}")
    return df


def load_csv(path):
    """Load OHLCV from CSV."""
    df = pd.read_csv(path)
    for col in ["timestamps", "timestamp", "date", "datetime", "time"]:
        if col in df.columns:
            df["timestamps"] = pd.to_datetime(df[col])
            break
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            print(f"CSV missing: {col}")
            sys.exit(1)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "amount" not in df.columns:
        df["amount"] = df["volume"] * df["close"]
    return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)


def infer_freq(df):
    if len(df) < 2:
        return "1h"
    delta = df["timestamps"].iloc[1] - df["timestamps"].iloc[0]
    seconds = int(delta.total_seconds())
    return {60: "1min", 300: "5min", 900: "15min", 3600: "1h", 14400: "4h", 86400: "1D"}.get(seconds, f"{seconds}s")


def get_signal(current_price, pred_df, threshold=0.5):
    avg_pred = pred_df["close"].mean()
    pct = (avg_pred - current_price) / current_price * 100
    if pct > threshold:
        return "BUY", pct
    elif pct < -threshold:
        return "SELL", pct
    return "HOLD", pct


def run_oos_backtest(df, predictor, lookback, forecast, step, capital, threshold, samples, verbose=False):
    """Walk-forward backtest. Returns trades, equity curve, final capital."""
    freq = infer_freq(df)
    n = len(df)
    max_start = n - forecast

    trades = []
    equity = []
    signals_log = []
    position = None
    cash = capital
    total_steps = 0

    i = lookback
    while i < max_start:
        total_steps += 1
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
        except Exception as e:
            if verbose:
                print(f"  Step {total_steps}: prediction failed ({e})")
            i += step
            continue

        # Log what actually happened
        actual_future = df.iloc[i:min(i + forecast, n)]
        actual_change = 0
        if len(actual_future) > 0:
            actual_end = actual_future["close"].iloc[-1]
            actual_change = (actual_end - current_price) / current_price * 100

        signals_log.append({
            "time": current_time,
            "price": current_price,
            "signal": signal,
            "pred_pct": pct_pred,
            "actual_pct": actual_change,
            "correct": (signal == "BUY" and actual_change > 0) or
                       (signal == "SELL" and actual_change < 0) or
                       (signal == "HOLD"),
        })

        if verbose:
            direction = "correct" if signals_log[-1]["correct"] else "WRONG"
            print(f"  Step {total_steps} | {current_time} | {signal} | pred: {pct_pred:+.2f}% | actual: {actual_change:+.2f}% | {direction}")

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
                "hold_bars": i - position["entry_idx"],
            })
            position = None

        equity.append({"time": current_time, "cash": cash, "price": current_price})
        i += step

    # Close open position
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
            "hold_bars": len(df) - 1 - position["entry_idx"],
        })

    return trades, equity, cash, total_steps, signals_log


def compute_metrics(trades, equity, initial_capital, final_capital, df, lookback):
    start_price = df["close"].iloc[lookback]
    end_price = df["close"].iloc[-1]
    bnh_return = (end_price - start_price) / start_price * 100
    total_return = (final_capital - initial_capital) / initial_capital * 100

    if not trades:
        return {
            "total_return": total_return, "buy_hold_return": bnh_return,
            "num_trades": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0,
            "max_drawdown": 0, "profit_factor": 0, "avg_hold": 0,
            "sharpe_approx": 0,
        }

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]

    win_rate = len(wins) / len(trades) * 100
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    avg_hold = np.mean([t["hold_bars"] for t in trades])

    gross_profit = sum(t["pnl_dollar"] for t in wins)
    gross_loss = abs(sum(t["pnl_dollar"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    peak = initial_capital
    max_dd = 0
    for point in equity:
        if point["cash"] > peak:
            peak = point["cash"]
        dd = (peak - point["cash"]) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Approximate Sharpe from trade returns
    returns = [t["pnl_pct"] for t in trades]
    sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

    return {
        "total_return": total_return, "buy_hold_return": bnh_return,
        "num_trades": len(trades), "win_rate": win_rate,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "max_drawdown": max_dd, "profit_factor": profit_factor,
        "avg_hold": avg_hold, "sharpe_approx": sharpe,
    }


def signal_accuracy(signals_log):
    """How often did the signal direction match reality?"""
    if not signals_log:
        return {}

    buy_signals = [s for s in signals_log if s["signal"] == "BUY"]
    sell_signals = [s for s in signals_log if s["signal"] == "SELL"]
    hold_signals = [s for s in signals_log if s["signal"] == "HOLD"]

    buy_correct = sum(1 for s in buy_signals if s["actual_pct"] > 0)
    sell_correct = sum(1 for s in sell_signals if s["actual_pct"] < 0)

    total_directional = len(buy_signals) + len(sell_signals)
    directional_correct = buy_correct + sell_correct

    # Mean absolute error of predictions
    pred_errors = [abs(s["pred_pct"] - s["actual_pct"]) for s in signals_log]

    return {
        "total_signals": len(signals_log),
        "buy_signals": len(buy_signals),
        "sell_signals": len(sell_signals),
        "hold_signals": len(hold_signals),
        "buy_accuracy": (buy_correct / len(buy_signals) * 100) if buy_signals else 0,
        "sell_accuracy": (sell_correct / len(sell_signals) * 100) if sell_signals else 0,
        "directional_accuracy": (directional_correct / total_directional * 100) if total_directional else 0,
        "mean_pred_error": np.mean(pred_errors),
    }


def print_validation_report(m, sig, trades, initial_capital, final_capital, total_steps,
                             data_range, test_range, threshold):
    print("\n" + "=" * 70)
    print("  KRONOS VALIDATION REPORT (OUT-OF-SAMPLE)")
    print("=" * 70)

    print(f"\n  DATA")
    print(f"  {'Full dataset:':<25} {data_range}")
    print(f"  {'Test period:':<25} {test_range}")
    print(f"  {'Steps evaluated:':<25} {total_steps}")
    print(f"  {'Signal threshold:':<25} {threshold}%")

    print(f"\n  SIGNAL ACCURACY")
    print(f"  {'Total signals:':<25} {sig.get('total_signals', 0)}")
    print(f"  {'BUY signals:':<25} {sig.get('buy_signals', 0)} (accuracy: {sig.get('buy_accuracy', 0):.1f}%)")
    print(f"  {'SELL signals:':<25} {sig.get('sell_signals', 0)} (accuracy: {sig.get('sell_accuracy', 0):.1f}%)")
    print(f"  {'HOLD signals:':<25} {sig.get('hold_signals', 0)}")
    print(f"  {'Directional accuracy:':<25} {sig.get('directional_accuracy', 0):.1f}%")
    print(f"  {'Mean prediction error:':<25} {sig.get('mean_pred_error', 0):.2f}%")

    print(f"\n  TRADING PERFORMANCE")
    print(f"  {'Trades executed:':<25} {m['num_trades']}")
    print(f"  {'Starting capital:':<25} ${initial_capital:,.2f}")
    print(f"  {'Final capital:':<25} ${final_capital:,.2f}")
    print(f"  {'Strategy return:':<25} {m['total_return']:+.2f}%")
    print(f"  {'Buy & hold return:':<25} {m['buy_hold_return']:+.2f}%")
    alpha = m['total_return'] - m['buy_hold_return']
    print(f"  {'Alpha:':<25} {alpha:+.2f}%")

    if m['num_trades'] > 0:
        print(f"\n  RISK METRICS")
        print(f"  {'Win rate:':<25} {m['win_rate']:.1f}%")
        print(f"  {'Avg win:':<25} {m['avg_win']:+.2f}%")
        print(f"  {'Avg loss:':<25} {m['avg_loss']:+.2f}%")
        print(f"  {'Profit factor:':<25} {m['profit_factor']:.2f}")
        print(f"  {'Max drawdown:':<25} {m['max_drawdown']:.2f}%")
        print(f"  {'Sharpe (approx):':<25} {m['sharpe_approx']:.2f}")
        print(f"  {'Avg hold (bars):':<25} {m['avg_hold']:.0f}")

    print("\n" + "-" * 70)

    # Verdict
    red_flags = []
    if m['num_trades'] < 10:
        red_flags.append(f"Only {m['num_trades']} trades - not statistically significant (need 20+)")
    if m['num_trades'] > 0 and m['win_rate'] == 100:
        red_flags.append("100% win rate - almost certainly overfitting or too few trades")
    if m['num_trades'] > 0 and m['win_rate'] > 80:
        red_flags.append(f"{m['win_rate']:.0f}% win rate is suspiciously high - verify with more data")
    if sig.get('directional_accuracy', 0) > 70:
        red_flags.append(f"{sig['directional_accuracy']:.0f}% directional accuracy exceeds what most quant funds achieve")
    if m['num_trades'] > 0 and m['max_drawdown'] == 0:
        red_flags.append("Zero drawdown - unrealistic, check for look-ahead bias")
    if sig.get('hold_signals', 0) > sig.get('total_signals', 1) * 0.9:
        red_flags.append("Model is HOLDing >90% of the time - threshold may be too high")

    if red_flags:
        print("  RED FLAGS:")
        for flag in red_flags:
            print(f"    - {flag}")
    else:
        print("  No obvious red flags detected.")

    green_flags = []
    if 10 <= m['num_trades'] <= 200:
        green_flags.append("Reasonable number of trades for statistical validity")
    if 45 <= m['win_rate'] <= 65 and m['num_trades'] >= 10:
        green_flags.append("Win rate in realistic range")
    if m['profit_factor'] > 1.0 and m['num_trades'] >= 10:
        green_flags.append("Profitable after costs (before slippage/fees)")
    if alpha > 0 and m['num_trades'] >= 10:
        green_flags.append("Outperforming buy & hold")

    if green_flags:
        print("\n  POSITIVES:")
        for flag in green_flags:
            print(f"    + {flag}")

    print("\n  IMPORTANT CAVEATS:")
    print("    - No transaction fees or slippage included")
    print("    - Real execution will differ from simulated fills")
    print("    - Past performance does not predict future results")
    print("    - Test on multiple time periods before trading real money")

    print("=" * 70 + "\n")


def run_threshold_sweep(df_test, predictor, lookback, forecast, step, capital, samples):
    """Test multiple thresholds to check if results are robust or fragile."""
    thresholds = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    print("\n" + "=" * 70)
    print("  THRESHOLD SWEEP (checking robustness)")
    print("=" * 70)
    print(f"  {'Thresh':>7} | {'Trades':>6} | {'Win%':>6} | {'Return':>8} | {'B&H':>8} | {'Alpha':>8} | {'PF':>6} | {'MaxDD':>6}")
    print(f"  {'-'*7} | {'-'*6} | {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*6} | {'-'*6}")

    for thresh in thresholds:
        trades, equity, final_cap, steps, _ = run_oos_backtest(
            df_test, predictor, lookback, forecast, step, capital, thresh, samples
        )
        m = compute_metrics(trades, equity, capital, final_cap, df_test, lookback)
        alpha = m['total_return'] - m['buy_hold_return']
        print(f"  {thresh:>6.2f}% | {m['num_trades']:>6} | {m['win_rate']:>5.1f}% | {m['total_return']:>+7.2f}% | {m['buy_hold_return']:>+7.2f}% | {alpha:>+7.2f}% | {m['profit_factor']:>5.2f} | {m['max_drawdown']:>5.2f}%")

    print("=" * 70)
    print("  If results flip wildly between thresholds, the edge is fragile.")
    print("  A robust strategy shows consistent alpha across a range.\n")


def main():
    parser = argparse.ArgumentParser(description="Kronos Out-of-Sample Validator")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    parser.add_argument("--bars", type=int, default=2000, help="Total bars to fetch")
    parser.add_argument("--csv", help="Use CSV instead of fetching")
    parser.add_argument("--test-pct", type=float, default=0.25, help="Fraction of data for out-of-sample test")
    parser.add_argument("--lookback", type=int, default=400, help="Context window")
    parser.add_argument("--forecast", type=int, default=24, help="Forecast horizon")
    parser.add_argument("--step", type=int, default=24, help="Step between predictions")
    parser.add_argument("--capital", type=float, default=10000, help="Starting capital")
    parser.add_argument("--threshold", type=float, default=0.5, help="Signal threshold (%%)")
    parser.add_argument("--samples", type=int, default=1, help="Forecast samples per step")
    parser.add_argument("--model", default="NeoQuasar/Kronos-small", help="Model name")
    parser.add_argument("--exchange", default=None, help="Exchange ID (kraken, kucoin, binanceus, coinbasepro)")
    parser.add_argument("--save-data", action="store_true", help="Save fetched data to CSV")
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print each signal")
    args = parser.parse_args()

    # Load model
    print("Loading Kronos model...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(args.model)
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    print("Model loaded.\n")

    # Get data
    if args.csv:
        print(f"Loading {args.csv}...")
        df = load_csv(args.csv)
    else:
        df = fetch_real_data(args.symbol, args.timeframe, args.bars, args.exchange)

    if args.save_data and not args.csv:
        fname = f"{args.symbol.replace('/', '_')}_{args.timeframe}_{len(df)}bars.csv"
        df.to_csv(fname, index=False)
        print(f"Saved data to {fname}")

    # Split into in-sample / out-of-sample
    n = len(df)
    split_idx = int(n * (1 - args.test_pct))

    # Make sure test set has enough bars for meaningful predictions
    # (lookback is prepended separately, so we only need forecast + a few steps)
    min_oos_bars = args.forecast + args.step * 3
    if n - split_idx < min_oos_bars:
        print(f"Test set too small ({n - split_idx} bars). Need at least {min_oos_bars}.")
        print("Fetch more data with --bars or reduce --test-pct.")
        sys.exit(1)

    if split_idx < args.lookback:
        print(f"Not enough in-sample data for lookback ({split_idx} < {args.lookback}).")
        print("Reduce --lookback or fetch more data.")
        sys.exit(1)

    df_test = df.iloc[split_idx - args.lookback:].reset_index(drop=True)

    data_range = f"{df['timestamps'].iloc[0].strftime('%Y-%m-%d')} to {df['timestamps'].iloc[-1].strftime('%Y-%m-%d')}"
    test_start = df['timestamps'].iloc[split_idx]
    test_range = f"{test_start.strftime('%Y-%m-%d')} to {df['timestamps'].iloc[-1].strftime('%Y-%m-%d')}"

    print(f"\nFull data:    {n} bars ({data_range})")
    print(f"In-sample:    {split_idx} bars (for tuning only, NOT tested)")
    print(f"Out-of-sample: {n - split_idx} bars ({test_range})")
    print(f"Test data (with lookback): {len(df_test)} bars\n")

    # Run out-of-sample backtest
    print("Running out-of-sample backtest...")
    trades, equity, final_capital, total_steps, signals_log = run_oos_backtest(
        df_test, predictor, args.lookback, args.forecast, args.step,
        args.capital, args.threshold, args.samples, args.verbose,
    )

    # Compute and display results
    m = compute_metrics(trades, equity, args.capital, final_capital, df_test, args.lookback)
    sig = signal_accuracy(signals_log)

    print_validation_report(
        m, sig, trades, args.capital, final_capital, total_steps,
        data_range, test_range, args.threshold,
    )

    # Trade log
    if trades:
        print("Trade log:")
        print(f"  {'#':>3}  {'Entry Time':>20}  {'Entry':>10}  {'Exit':>10}  {'P&L':>8}  {'Bars':>5}")
        print(f"  {'---':>3}  {'--------------------':>20}  {'--------':>10}  {'--------':>10}  {'------':>8}  {'----':>5}")
        for i, t in enumerate(trades, 1):
            etime = t['entry_time'].strftime('%Y-%m-%d %H:%M') if hasattr(t['entry_time'], 'strftime') else str(t['entry_time'])
            print(f"  {i:3d}  {etime:>20}  ${t['entry_price']:>9,.2f}  ${t['exit_price']:>9,.2f}  {t['pnl_pct']:>+7.2f}%  {t['hold_bars']:>5d}")

    # Threshold sweep
    if args.sweep:
        run_threshold_sweep(df_test, predictor, args.lookback, args.forecast, args.step, args.capital, args.samples)


if __name__ == "__main__":
    main()
