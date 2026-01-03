"""
QuantConnect QuantBook - Run Regression Strategy & Export Results

This script runs the exact regression strategy from main.py in QuantBook
and exports all trade results in chunks for download.

Usage:
    1. Create a new Research notebook in QuantConnect
    2. Copy this script into cells
    3. Run to backtest the strategy
    4. Download the chunked results from Object Store

Output Format (matches training data format):
    - strategy_results_chunk_N.pkl - Trade results with predictions
    - strategy_metadata.json - Summary statistics
"""

# ============================================================
# QUANTCONNECT RESEARCH NOTEBOOK CODE
# Copy everything below into a QuantConnect research notebook
# ============================================================

from AlgorithmImports import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import time
import json
import pickle

# Initialize QuantBook
qb = QuantBook()

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    # Date range (matches historical backtest)
    'start_date': datetime(2010, 1, 1),
    'end_date': datetime(2025, 12, 31),

    # Strategy parameters (from main.py - MUST MATCH EXACTLY)
    'hold_duration_days': 3,
    'training_lookback_years': 4,  # main.py uses 4 years, NOT 30 days
    'min_training_samples': 5,
    'position_size_pct': 0.25,  # 25% of portfolio per trade

    # Price data resolution
    'price_resolution': Resolution.Hour,
    'hours_after_split': 72,  # 3 days of hourly data

    # Sector ETF for momentum (MUST USE in regression like main.py)
    'sector_etf': 'XLK',
    'sector_roc_period': 22,

    # Rate limiting
    'batch_size': 50,
    'batch_delay_seconds': 1,

    # Sectors to include (main.py ONLY uses TECHNOLOGY)
    'include_sectors': [
        MorningstarSectorCode.TECHNOLOGY,
    ],

    # Output chunking (QC Object Store has limits)
    'chunk_size': 1000,  # Episodes per chunk
    'output_prefix': 'strategy_results',
}

print("="*60)
print("Regression Strategy Backtest & Export")
print("="*60)
print(f"Date range: {CONFIG['start_date'].date()} to {CONFIG['end_date'].date()}")


# ============================================================
# STEP 1: BUILD UNIVERSE
# ============================================================

print("\n[Step 1] Building universe...")

all_symbols_qc = []
symbol_map = {}

def sector_filter(fundamental):
    """Filter for target sectors."""
    return [
        x.symbol for x in fundamental
        if (x.asset_classification.morningstar_sector_code in CONFIG['include_sectors']
            and x.has_fundamental_data
            and x.price > 1)
    ]

universe_history = qb.universe_history(
    Fundamentals,
    CONFIG['start_date'],
    CONFIG['end_date'],
    sector_filter
)

all_symbols = set()
for date_symbols in universe_history:
    for symbol in date_symbols:
        ticker = str(symbol).split()[0] if ' ' in str(symbol) else str(symbol.value)
        all_symbols.add(ticker)

print(f"  Found {len(all_symbols)} unique stocks")

# Add symbols to QuantBook
for ticker in all_symbols:
    try:
        equity = qb.add_equity(ticker, Resolution.Daily)
        all_symbols_qc.append(equity.symbol)
        symbol_map[ticker] = equity.symbol
    except:
        pass

print(f"  Added {len(all_symbols_qc)} symbols")

# Add sector ETF and calculate ROC history (MATCHES main.py)
print(f"\n  Adding sector ETF: {CONFIG['sector_etf']}...")
sector_etf = qb.add_equity(CONFIG['sector_etf'], Resolution.Daily)
sector_etf_prices = qb.history(
    sector_etf.symbol,
    CONFIG['start_date'] - timedelta(days=CONFIG['sector_roc_period'] + 60),
    CONFIG['end_date'] + timedelta(days=10),
    Resolution.Daily
)['close'].droplevel(0)

# Calculate 22-day ROC for sector ETF (matches main.py)
sector_roc_series = sector_etf_prices.pct_change(CONFIG['sector_roc_period'])
print(f"  Sector ROC calculated: {len(sector_roc_series)} data points")


def get_sector_roc_at_date(date):
    """Get sector ROC at a specific date (matches main.py logic)."""
    try:
        # Get most recent ROC value at or before this date
        filtered = sector_roc_series[sector_roc_series.index <= date]
        if filtered.empty:
            return 0.0
        return float(filtered.iloc[-1])
    except:
        return 0.0


# ============================================================
# STEP 2: GET ALL SPLIT EVENTS
# ============================================================

print("\n[Step 2] Fetching split events...")

def get_splits_batched(qb, symbols, start_date, end_date, batch_size=50):
    """Get splits with batching."""
    all_splits = []
    symbol_list = list(symbols)

    for i in range(0, len(symbol_list), batch_size):
        batch = symbol_list[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(symbol_list) + batch_size - 1) // batch_size

        print(f"  Batch {batch_num}/{total_batches}...", end='')

        try:
            splits = qb.history(Split, batch, start_date, end_date)
            if not splits.empty:
                all_splits.append(splits)
                print(f" {len(splits)} splits")
            else:
                print(" 0 splits")
        except Exception as e:
            print(f" Error: {e}")

        if i + batch_size < len(symbol_list):
            time.sleep(CONFIG['batch_delay_seconds'])

    if all_splits:
        return pd.concat(all_splits)
    return pd.DataFrame()

splits_df = get_splits_batched(
    qb, all_symbols_qc,
    CONFIG['start_date'], CONFIG['end_date'],
    CONFIG['batch_size']
)

# Filter to warnings only (tradeable signals)
if not splits_df.empty:
    warning_splits = splits_df[splits_df['type'] == SplitType.WARNING]
    print(f"  Total tradeable splits: {len(warning_splits)}")
else:
    warning_splits = pd.DataFrame()
    print("  No splits found!")


# ============================================================
# STEP 3: GET HISTORICAL PRICE DATA FOR REGRESSION
# ============================================================

print("\n[Step 3] Fetching historical data for regression model...")

# We need to build a rolling regression model
# Get daily close prices for all symbols
daily_prices = {}

for i, symbol in enumerate(all_symbols_qc):
    if i % 100 == 0:
        print(f"  Fetching prices: {i}/{len(all_symbols_qc)}...")

    try:
        hist = qb.history(
            symbol,
            CONFIG['start_date'] - timedelta(days=60),
            CONFIG['end_date'] + timedelta(days=10),
            Resolution.Daily
        )
        if not hist.empty:
            daily_prices[str(symbol)] = hist['close'].droplevel(0)
    except:
        pass

    if i % CONFIG['batch_size'] == 0 and i > 0:
        time.sleep(0.5)

print(f"  Got daily prices for {len(daily_prices)} symbols")


# ============================================================
# STEP 4: RUN REGRESSION STRATEGY ON EACH SPLIT
# ============================================================

print("\n[Step 4] Running regression strategy...")

def get_3day_return(prices_series, date, hold_days=3):
    """Calculate 3-day return starting from date."""
    try:
        # Find the entry date (next trading day after date)
        future_dates = prices_series.index[prices_series.index > date]
        if len(future_dates) < hold_days + 1:
            return None

        entry_date = future_dates[0]
        exit_date = future_dates[hold_days]

        entry_price = prices_series.loc[entry_date]
        exit_price = prices_series.loc[exit_date]

        return (exit_price - entry_price) / entry_price
    except:
        return None

def build_regression_features(splits_history, daily_prices_dict, target_date, lookback_years=4):
    """
    Build regression training data from historical splits.

    MATCHES main.py EXACTLY:
    Features: [split_factor, sector_roc]  (2 features, NOT 1!)
    Target: 3-day return after split
    """
    X = []
    y = []

    # Use years for lookback, matching main.py's training_lookback_years
    lookback_start = target_date - timedelta(days=lookback_years * 365)

    for _, row in splits_history.iterrows():
        split_time = row['time']
        if split_time < lookback_start or split_time >= target_date:
            continue

        symbol_str = str(row['symbol'])
        split_factor = row.get('splitfactor', row.get('value', 1.0))

        if symbol_str not in daily_prices_dict:
            continue

        prices = daily_prices_dict[symbol_str]
        ret = get_3day_return(prices, split_time, CONFIG['hold_duration_days'])

        if ret is not None and not np.isnan(ret) and abs(ret) < 2.0:  # Filter outliers
            # Get sector ROC at the split time (MATCHES main.py)
            sector_roc = get_sector_roc_at_date(split_time)

            # 2 features: [split_factor, sector_roc] - MATCHES main.py
            X.append([split_factor, sector_roc])
            y.append(ret)

    return np.array(X), np.array(y)

def predict_return(model, split_factor, sector_roc):
    """Predict return using fitted model (MATCHES main.py - uses 2 features)."""
    try:
        # 2 features: [split_factor, sector_roc] - MATCHES main.py
        return model.predict([[split_factor, sector_roc]])[0]
    except:
        return 0.0

# Process each split and generate predictions
results = []
processed = 0
skipped = 0

if not warning_splits.empty:
    splits_reset = warning_splits.reset_index()
    total_splits = len(splits_reset)

    print(f"  Processing {total_splits} splits...")

    for idx, row in splits_reset.iterrows():
        symbol = row['symbol']
        symbol_str = str(symbol)
        split_time = row['time']
        split_factor = row.get('splitfactor', row.get('value', 1.0))

        processed += 1
        if processed % 100 == 0:
            print(f"  Processed {processed}/{total_splits}...")

        # Build regression model from historical data (MATCHES main.py - 4 year lookback)
        X, y = build_regression_features(
            splits_reset[splits_reset['time'] < split_time],
            daily_prices,
            split_time,
            CONFIG['training_lookback_years']
        )

        if len(X) < CONFIG['min_training_samples']:
            skipped += 1
            continue

        # Fit regression (2 features: split_factor, sector_roc)
        model = LinearRegression()
        model.fit(X, y)

        # Get current sector ROC for prediction (MATCHES main.py)
        current_sector_roc = get_sector_roc_at_date(split_time)

        # Make prediction using both features (MATCHES main.py)
        predicted_return = predict_return(model, split_factor, current_sector_roc)

        # MATCHES main.py: Skip if predicted_return == 0
        if predicted_return == 0:
            skipped += 1
            continue

        # Determine trade direction based on prediction (MATCHES main.py: np.sign)
        direction = 1.0 if predicted_return > 0 else -1.0

        # Get actual price trajectory
        if symbol_str not in daily_prices:
            skipped += 1
            continue

        prices = daily_prices[symbol_str]

        # Get entry price and actual return
        future_dates = prices.index[prices.index > split_time]
        if len(future_dates) < CONFIG['hold_duration_days'] + 1:
            skipped += 1
            continue

        entry_date = future_dates[0].to_pydatetime()  # Convert to Python datetime
        exit_date = future_dates[CONFIG['hold_duration_days']].to_pydatetime()

        entry_price = float(prices.iloc[prices.index.get_loc(future_dates[0])])
        exit_price = float(prices.iloc[prices.index.get_loc(future_dates[CONFIG['hold_duration_days']])])

        # Calculate actual return based on direction
        if direction > 0:
            actual_return = (exit_price - entry_price) / entry_price
        else:
            actual_return = (entry_price - exit_price) / entry_price

        # Get hourly price trajectory from ENTRY DATE (not split_time)
        # This gives us the actual intra-trade price movements
        price_trajectory = []
        try:
            # Fetch hourly data starting from entry, extending past exit
            traj_start = entry_date
            traj_end = entry_date + timedelta(hours=CONFIG['hours_after_split'] + 24)  # Extra buffer

            hourly_hist = qb.history(
                symbol,
                traj_start,
                traj_end,
                Resolution.Hour
            )

            if not hourly_hist.empty:
                traj_prices = hourly_hist['close'].droplevel(0)
                price_trajectory = traj_prices.values.tolist()

                # Log if trajectory is shorter than expected
                if len(price_trajectory) < 24:
                    print(f"    WARNING: {symbol_str} short trajectory: {len(price_trajectory)} hours")
        except Exception as e:
            print(f"    ERROR fetching hourly data for {symbol_str}: {e}")
            price_trajectory = []

        # Skip episodes with insufficient trajectory data for RL training
        if len(price_trajectory) < 24:
            skipped += 1
            continue

        # Build result record (matches training data format)
        result = {
            'symbol': symbol_str,
            'split_time': split_time.isoformat(),
            'split_factor': float(split_factor),
            'direction': float(direction),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_date': entry_date.isoformat(),
            'exit_date': exit_date.isoformat(),
            'hold_duration_hours': CONFIG['hold_duration_days'] * 24,
            'predicted_return': float(predicted_return),
            'actual_return': float(actual_return),
            'final_pnl_pct': float(actual_return),  # Matches training format
            'is_penny_stock': bool(entry_price < 5.0),
            'sector_roc': float(current_sector_roc),  # Actual sector ROC at trade time
            'regression_r2': float(model.score(X, y)) if len(X) > 1 else 0.0,
            'training_samples': len(X),
            # Model has 2 coefficients: [split_factor_coef, sector_roc_coef]
            'model_coef_split': float(model.coef_[0]),
            'model_coef_sector': float(model.coef_[1]) if len(model.coef_) > 1 else 0.0,
            'model_intercept': float(model.intercept_),
            'price_trajectory': price_trajectory,
        }

        results.append(result)

        # Rate limiting
        if processed % CONFIG['batch_size'] == 0:
            time.sleep(CONFIG['batch_delay_seconds'])

print(f"  Processed: {processed}")
print(f"  Skipped (insufficient data): {skipped}")
print(f"  Valid results: {len(results)}")


# ============================================================
# STEP 5: EXPORT IN CHUNKS
# ============================================================

print("\n[Step 5] Exporting results in chunks...")

chunk_size = CONFIG['chunk_size']
num_chunks = (len(results) + chunk_size - 1) // chunk_size

for chunk_idx in range(num_chunks):
    start_idx = chunk_idx * chunk_size
    end_idx = min((chunk_idx + 1) * chunk_size, len(results))
    chunk_data = results[start_idx:end_idx]

    filename = f"{CONFIG['output_prefix']}_chunk_{chunk_idx}.pkl"
    pickle_bytes = pickle.dumps(chunk_data)
    qb.object_store.save_bytes(filename, pickle_bytes)
    print(f"  Saved: {filename} ({len(chunk_data)} episodes, {len(pickle_bytes)} bytes)")

# Save metadata
results_df = pd.DataFrame(results)

# Calculate trajectory statistics
traj_lengths = [len(r['price_trajectory']) for r in results] if results else [0]

metadata = {
    'export_date': datetime.now().isoformat(),
    'config': {k: str(v) for k, v in CONFIG.items()},
    'total_episodes': len(results),
    'num_chunks': num_chunks,
    'chunk_size': chunk_size,
    'unique_symbols': len(results_df['symbol'].unique()) if len(results_df) > 0 else 0,
    'date_range': {
        'start': results_df['split_time'].min() if len(results_df) > 0 else '',
        'end': results_df['split_time'].max() if len(results_df) > 0 else '',
    },
    'statistics': {
        'long_trades': len(results_df[results_df['direction'] == 1.0]) if len(results_df) > 0 else 0,
        'short_trades': len(results_df[results_df['direction'] == -1.0]) if len(results_df) > 0 else 0,
        'avg_predicted_return': float(results_df['predicted_return'].mean()) if len(results_df) > 0 else 0,
        'avg_actual_return': float(results_df['actual_return'].mean()) if len(results_df) > 0 else 0,
        'prediction_correlation': float(results_df['predicted_return'].corr(results_df['actual_return'])) if len(results_df) > 0 else 0,
        'win_rate': float((results_df['actual_return'] > 0).mean()) if len(results_df) > 0 else 0,
        'avg_r2': float(results_df['regression_r2'].mean()) if len(results_df) > 0 else 0,
    },
    'trajectory_stats': {
        'min_length': int(min(traj_lengths)),
        'max_length': int(max(traj_lengths)),
        'mean_length': float(np.mean(traj_lengths)),
        'gte_24h': sum(1 for l in traj_lengths if l >= 24),
        'gte_48h': sum(1 for l in traj_lengths if l >= 48),
        'gte_72h': sum(1 for l in traj_lengths if l >= 72),
    }
}

metadata_string = json.dumps(metadata, indent=2)
qb.object_store.save(f"{CONFIG['output_prefix']}_metadata.json", metadata_string)
print(f"  Saved: {CONFIG['output_prefix']}_metadata.json")


# ============================================================
# STEP 6: SUMMARY STATISTICS
# ============================================================

print("\n" + "="*60)
print("STRATEGY BACKTEST COMPLETE")
print("="*60)

if len(results_df) > 0:
    print(f"\nResults Summary:")
    print(f"  Total trades: {len(results_df)}")
    print(f"  Unique symbols: {len(results_df['symbol'].unique())}")
    print(f"  Date range: {results_df['split_time'].min()[:10]} to {results_df['split_time'].max()[:10]}")

    print(f"\n  Direction Breakdown:")
    print(f"    Long trades: {len(results_df[results_df['direction'] == 1.0])}")
    print(f"    Short trades: {len(results_df[results_df['direction'] == -1.0])}")

    print(f"\n  Prediction Quality:")
    print(f"    Avg predicted return: {results_df['predicted_return'].mean():.4f}")
    print(f"    Avg actual return: {results_df['actual_return'].mean():.4f}")
    corr = results_df['predicted_return'].corr(results_df['actual_return'])
    print(f"    Prediction correlation: {corr:.4f}")
    print(f"    Win rate: {(results_df['actual_return'] > 0).mean():.2%}")

    print(f"\n  Sector ROC Statistics (XLK 22-day):")
    print(f"    Mean sector ROC: {results_df['sector_roc'].mean():.4f}")
    print(f"    Std sector ROC: {results_df['sector_roc'].std():.4f}")

    print(f"\n  Regression Model (2 features: split_factor, sector_roc):")
    print(f"    Avg split_factor coef: {results_df['model_coef_split'].mean():.4f}")
    print(f"    Avg sector_roc coef: {results_df['model_coef_sector'].mean():.4f}")
    print(f"    Avg intercept: {results_df['model_intercept'].mean():.4f}")
    print(f"    Avg R²: {results_df['regression_r2'].mean():.4f}")

    print(f"\n  P&L Statistics:")
    print(f"    Mean return: {results_df['actual_return'].mean():.4f}")
    print(f"    Std return: {results_df['actual_return'].std():.4f}")
    print(f"    Sharpe (annualized): {results_df['actual_return'].mean() / results_df['actual_return'].std() * np.sqrt(252/3):.2f}")

    # Trajectory statistics (critical for RL training)
    # traj_lengths already computed above for metadata
    print(f"\n  Price Trajectory Statistics (for RL training):")
    print(f"    Min length: {min(traj_lengths)} hours")
    print(f"    Max length: {max(traj_lengths)} hours")
    print(f"    Mean length: {np.mean(traj_lengths):.1f} hours")
    print(f"    >= 24 hours: {sum(1 for l in traj_lengths if l >= 24)} ({sum(1 for l in traj_lengths if l >= 24)/len(traj_lengths):.1%})")
    print(f"    >= 48 hours: {sum(1 for l in traj_lengths if l >= 48)} ({sum(1 for l in traj_lengths if l >= 48)/len(traj_lengths):.1%})")

    # Long vs Short breakdown
    long_df = results_df[results_df['direction'] == 1.0]
    short_df = results_df[results_df['direction'] == -1.0]

    if len(long_df) > 0:
        print(f"\n  Long Trades:")
        print(f"    Count: {len(long_df)}")
        print(f"    Mean return: {long_df['actual_return'].mean():.4f}")
        print(f"    Win rate: {(long_df['actual_return'] > 0).mean():.2%}")

    if len(short_df) > 0:
        print(f"\n  Short Trades:")
        print(f"    Count: {len(short_df)}")
        print(f"    Mean return: {short_df['actual_return'].mean():.4f}")
        print(f"    Win rate: {(short_df['actual_return'] > 0).mean():.2%}")

print("\n" + "="*60)
print("FILES SAVED TO OBJECT STORE")
print("="*60)
print(f"\nChunk files ({num_chunks} total):")
for i in range(num_chunks):
    print(f"  - {CONFIG['output_prefix']}_chunk_{i}.pkl")
print(f"  - {CONFIG['output_prefix']}_metadata.json")

print("\nTo download:")
print("  1. Go to: https://www.quantconnect.com/terminal/#organization/object-store")
print("  2. Download all chunk files")
print("  3. Use reassemble_chunks.py to combine them locally")
print("="*60)
