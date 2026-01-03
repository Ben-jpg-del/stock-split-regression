"""
QuantConnect QuantBook Data Export for SAC Risk Management Training

This script exports high-quality historical data from QuantConnect for
training the SAC risk management agent locally.

Usage:
    1. Create a new Research notebook in QuantConnect
    2. Copy this script into a cell
    3. Run to export data
    4. Download the generated CSV files

The exported data includes:
    - All stock splits in the technology sector (2015-2025)
    - Hourly price trajectories around each split event
    - Sector momentum (XLK ROC) at time of split
    - Actual P&L outcomes for 3-day hold periods
"""

# ============================================================
# QUANTCONNECT RESEARCH NOTEBOOK CODE
# Copy everything below into a QuantConnect research notebook
# ============================================================

from AlgorithmImports import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

# Initialize QuantBook
qb = QuantBook()

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    # Date range for data export (EXPANDED)
    'start_date': datetime(2010, 1, 1),  # Extended back to 2010
    'end_date': datetime(2025, 12, 31),

    # Hold duration (matches strategy)
    'hold_duration_days': 3,

    # Price data resolution
    'price_resolution': Resolution.Hour,

    # Hours of price data to capture per episode
    'hours_before_split': 24,   # 1 day before
    'hours_after_split': 72,    # 3 days after (hold period)

    # Sector ETF for momentum
    'sector_etf': 'XLK',
    'sector_roc_period': 22,  # Days for ROC calculation

    # Rate limiting (QC internal limits)
    'batch_size': 50,           # Symbols per batch
    'batch_delay_seconds': 1,   # Delay between batches

    # EXPANDED: Include multiple sectors for more data
    'include_sectors': [
        MorningstarSectorCode.TECHNOLOGY,
        MorningstarSectorCode.COMMUNICATION_SERVICES,
        MorningstarSectorCode.CONSUMER_CYCLICAL,
        MorningstarSectorCode.HEALTHCARE,
        MorningstarSectorCode.FINANCIAL_SERVICES,
    ],

    # EXPANDED: Include both split types
    'include_split_occurred': True,  # Include actual split events too

    # EXPANDED: Data augmentation - create multiple episodes per split
    'augment_time_offsets': [-12, -6, 0, 6, 12],  # Hours offset from split time
    'augment_hold_durations': [24, 48, 72],  # Different hold periods in hours

    # Output files
    'output_episodes': 'training_episodes.csv',
    'output_prices': 'price_trajectories.pkl',
    'output_metadata': 'export_metadata.json',
}

print("="*60)
print("QuantConnect Data Export for SAC Training")
print("="*60)
print(f"Date range: {CONFIG['start_date'].date()} to {CONFIG['end_date'].date()}")
print(f"Hold duration: {CONFIG['hold_duration_days']} days")


# ============================================================
# STEP 1: GET MULTI-SECTOR UNIVERSE (EXPANDED)
# ============================================================

print("\n[Step 1] Building multi-sector universe...")

all_symbols_qc = []
symbol_map = {}

def multi_sector_filter(fundamental):
    """Filter for multiple sectors - EXPANDED for more data"""
    target_sectors = CONFIG['include_sectors']
    return [
        x.symbol for x in fundamental
        if (x.asset_classification.morningstar_sector_code in target_sectors
            and x.has_fundamental_data
            and x.price > 1)
    ]

print(f"  Target sectors: {len(CONFIG['include_sectors'])}")
print("  Fetching multi-sector universe via universe_history...")

universe_history = qb.universe_history(
    Fundamentals,
    CONFIG['start_date'],
    CONFIG['end_date'],
    multi_sector_filter
)

all_symbols = set()
for date_symbols in universe_history:
    for symbol in date_symbols:
        ticker = str(symbol).split()[0] if ' ' in str(symbol) else str(symbol.value)
        all_symbols.add(ticker)

print(f"  Found {len(all_symbols)} unique stocks from universe history")

# Add all symbols to QuantBook
added_count = 0
for ticker in all_symbols:
    try:
        equity = qb.add_equity(ticker, Resolution.Daily)
        all_symbols_qc.append(equity.symbol)
        symbol_map[ticker] = equity.symbol
        added_count += 1
    except Exception as e:
        pass  # Skip symbols that can't be added

print(f"  Successfully added {added_count} symbols to QuantBook")


# ============================================================
# STEP 2: GET ALL SPLIT EVENTS (EXPANDED)
# ============================================================

print("\n[Step 2] Fetching split events...")

def get_splits_for_symbols(qb, symbols, start_date, end_date, batch_size=50):
    """Get all split events for given symbols with rate limiting."""
    all_splits = []
    symbol_list = list(symbols)

    for i in range(0, len(symbol_list), batch_size):
        batch = symbol_list[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(symbol_list) + batch_size - 1) // batch_size

        print(f"  Batch {batch_num}/{total_batches}: {len(batch)} symbols...", end='')

        try:
            # Get splits for this batch
            splits = qb.history(Split, batch, start_date, end_date)

            if not splits.empty:
                all_splits.append(splits)
                print(f" Found {len(splits)} splits")
            else:
                print(" No splits")

        except Exception as e:
            print(f" Error: {e}")

        # Rate limiting delay
        if i + batch_size < len(symbol_list):
            time.sleep(CONFIG['batch_delay_seconds'])

    if all_splits:
        return pd.concat(all_splits)
    return pd.DataFrame()

splits_df = get_splits_for_symbols(
    qb,
    all_symbols_qc,  # EXPANDED: Use all symbols, not just tech
    CONFIG['start_date'],
    CONFIG['end_date'],
    CONFIG['batch_size']
)

print(f"  Total split events: {len(splits_df)}")

# EXPANDED: Include both split types for more data
if not splits_df.empty:
    # The splits dataframe has multi-index (symbol, time)
    warning_splits = splits_df[splits_df['type'] == SplitType.WARNING]
    print(f"  Split warnings (tradeable): {len(warning_splits)}")

    if CONFIG['include_split_occurred']:
        occurred_splits = splits_df[splits_df['type'] == SplitType.SPLIT_OCCURRED]
        print(f"  Split occurred events: {len(occurred_splits)}")
        # Combine both types
        all_tradeable_splits = pd.concat([warning_splits, occurred_splits])
        print(f"  Total tradeable splits: {len(all_tradeable_splits)}")
    else:
        all_tradeable_splits = warning_splits
else:
    all_tradeable_splits = pd.DataFrame()
    print("  No splits found!")


# ============================================================
# STEP 3: GET SECTOR ETF DATA FOR ROC CALCULATION
# ============================================================

print("\n[Step 3] Fetching sector ETF data...")

sector_symbol = qb.add_equity(CONFIG['sector_etf'], Resolution.Daily).symbol

sector_history = qb.history(
    sector_symbol,
    CONFIG['start_date'] - timedelta(days=CONFIG['sector_roc_period'] + 10),
    CONFIG['end_date'],
    Resolution.Daily
)

print(f"  XLK daily bars: {len(sector_history)}")

# Calculate ROC series
if not sector_history.empty:
    sector_close = sector_history['close'].droplevel(0)  # Remove symbol level
    sector_roc = sector_close.pct_change(CONFIG['sector_roc_period'])
    print(f"  ROC values calculated: {len(sector_roc.dropna())}")
else:
    sector_roc = pd.Series()
    print("  Warning: No sector data!")


# ============================================================
# STEP 4: BUILD TRAINING EPISODES (EXPANDED WITH AUGMENTATION)
# ============================================================

print("\n[Step 4] Building training episodes with augmentation...")

def get_price_trajectory(qb, symbol, start_time, end_time, resolution=Resolution.Hour):
    """Get hourly price data for a symbol between times."""
    try:
        history = qb.history(
            symbol,
            start_time,
            end_time,
            resolution
        )
        if not history.empty:
            return history['close'].droplevel(0).values  # Remove symbol level
        return None
    except Exception:
        return None

def get_sector_roc_at_date(sector_roc_series, date):
    """Get sector ROC value closest to given date."""
    if sector_roc_series.empty:
        return 0.0

    # Find closest date on or before
    mask = sector_roc_series.index <= date
    if mask.any():
        return sector_roc_series[mask].iloc[-1]
    return 0.0

def create_episode(symbol, split_time, split_factor, prices, entry_idx, hold_hours, sector_roc_value, direction):
    """Create a single episode dict."""
    exit_idx = min(entry_idx + hold_hours, len(prices) - 1)
    if exit_idx <= entry_idx:
        return None

    entry_price = prices[entry_idx]
    exit_price = prices[exit_idx]

    if direction > 0:
        pnl = (exit_price - entry_price) / entry_price
    else:
        pnl = (entry_price - exit_price) / entry_price

    return {
        'symbol': str(symbol),
        'split_time': split_time.isoformat(),
        'split_factor': float(split_factor),
        'direction': float(direction),
        'entry_price': float(entry_price),
        'exit_price': float(exit_price),
        'entry_idx': int(entry_idx),
        'exit_idx': int(exit_idx),
        'hold_duration_hours': int(exit_idx - entry_idx),
        'sector_roc': float(sector_roc_value),
        'is_penny_stock': bool(entry_price < 5.0),
        'final_pnl_pct': float(pnl),
        'price_trajectory': prices.tolist(),
    }

# Process each split event with augmentation
episodes = []
failed_count = 0
processed_count = 0
augmented_count = 0

time_offsets = CONFIG.get('augment_time_offsets', [0])
hold_durations = CONFIG.get('augment_hold_durations', [72])

if not all_tradeable_splits.empty:
    # Reset index to access symbol and time
    splits_reset = all_tradeable_splits.reset_index()
    total_splits = len(splits_reset)

    print(f"  Processing {total_splits} splits with {len(time_offsets)} offsets x {len(hold_durations)} durations x 2 directions")
    print(f"  Max episodes per split: {len(time_offsets) * len(hold_durations) * 2}")

    for idx, row in splits_reset.iterrows():
        symbol = row['symbol']
        split_time = row['time']
        split_factor = row['splitfactor'] if 'splitfactor' in row else row.get('value', 1.0)

        processed_count += 1
        if processed_count % 50 == 0:
            print(f"  Processing {processed_count}/{total_splits}... ({len(episodes)} episodes so far)")

        # Get extended price trajectory for augmentation
        # Need extra data before and after for different offsets
        max_offset = max(abs(o) for o in time_offsets) if time_offsets else 0
        max_hold = max(hold_durations) if hold_durations else 72

        trajectory_start = split_time - timedelta(hours=CONFIG['hours_before_split'] + max_offset)
        trajectory_end = split_time + timedelta(hours=max_hold + max_offset + 24)

        # Get price trajectory
        prices = get_price_trajectory(
            qb, symbol, trajectory_start, trajectory_end, CONFIG['price_resolution']
        )

        if prices is None or len(prices) < 24:
            failed_count += 1
            continue

        # Get sector ROC at split time
        sector_roc_value = get_sector_roc_at_date(sector_roc, split_time)

        # AUGMENTATION: Create episodes with different time offsets and hold durations
        for time_offset in time_offsets:
            # Adjust entry index based on offset
            base_entry_idx = CONFIG['hours_before_split'] + max_offset + time_offset

            if base_entry_idx < 0 or base_entry_idx >= len(prices):
                continue

            for hold_hours in hold_durations:
                for direction in [1.0, -1.0]:
                    ep = create_episode(
                        symbol, split_time, split_factor, prices,
                        base_entry_idx, hold_hours, sector_roc_value, direction
                    )
                    if ep:
                        episodes.append(ep)
                        augmented_count += 1

        # Rate limiting
        if processed_count % CONFIG['batch_size'] == 0:
            time.sleep(CONFIG['batch_delay_seconds'])

print(f"  Splits processed: {processed_count}")
print(f"  Failed (insufficient data): {failed_count}")
print(f"  Total episodes created: {len(episodes)}")
print(f"  Augmentation factor: {len(episodes) / max(1, processed_count - failed_count):.1f}x")


# ============================================================
# STEP 5: EXPORT DATA VIA OBJECT STORE
# ============================================================

print("\n[Step 5] Exporting data via Object Store (downloadable)...")

# Convert to DataFrame for easy export
episodes_df = pd.DataFrame(episodes)

# Build metadata
metadata = {
    'export_date': datetime.now().isoformat(),
    'config': {k: str(v) for k, v in CONFIG.items()},
    'total_episodes': len(episodes),
    'unique_symbols': len(episodes_df['symbol'].unique()) if len(episodes_df) > 0 else 0,
    'date_range': {
        'start': CONFIG['start_date'].isoformat(),
        'end': CONFIG['end_date'].isoformat(),
    },
    'statistics': {
        'long_episodes': len(episodes_df[episodes_df['direction'] == 1.0]) if len(episodes_df) > 0 else 0,
        'short_episodes': len(episodes_df[episodes_df['direction'] == -1.0]) if len(episodes_df) > 0 else 0,
        'penny_stock_episodes': len(episodes_df[episodes_df['is_penny_stock'] == True]) if len(episodes_df) > 0 else 0,
        'avg_pnl_long': float(episodes_df[episodes_df['direction'] == 1.0]['final_pnl_pct'].mean()) if len(episodes_df) > 0 else 0,
        'avg_pnl_short': float(episodes_df[episodes_df['direction'] == -1.0]['final_pnl_pct'].mean()) if len(episodes_df) > 0 else 0,
    }
}

# Save to Object Store (this makes files downloadable!)
import pickle
import io

# 1. Save CSV summary to Object Store
episodes_summary = episodes_df.drop(columns=['price_trajectory'])
csv_string = episodes_summary.to_csv(index=False)
qb.object_store.save(CONFIG['output_episodes'], csv_string)
print(f"  Saved to Object Store: {CONFIG['output_episodes']} ({len(episodes_summary)} rows)")

# 2. Save pickle with full price trajectories to Object Store
pickle_bytes = pickle.dumps(episodes)
qb.object_store.save_bytes(CONFIG['output_prices'], pickle_bytes)
print(f"  Saved to Object Store: {CONFIG['output_prices']} ({len(pickle_bytes)} bytes)")

# 3. Save metadata JSON to Object Store
metadata_string = json.dumps(metadata, indent=2)
qb.object_store.save(CONFIG['output_metadata'], metadata_string)
print(f"  Saved to Object Store: {CONFIG['output_metadata']}")


# ============================================================
# STEP 6: SUMMARY STATISTICS
# ============================================================

print("\n" + "="*60)
print("EXPORT COMPLETE")
print("="*60)

if len(episodes_df) > 0:
    print(f"\nDataset Statistics:")
    print(f"  Total episodes: {len(episodes_df)}")
    print(f"  Unique symbols: {len(episodes_df['symbol'].unique())}")
    print(f"  Date range: {episodes_df['split_time'].min()[:10]} to {episodes_df['split_time'].max()[:10]}")
    print(f"\n  Long episodes: {len(episodes_df[episodes_df['direction'] == 1.0])}")
    print(f"  Short episodes: {len(episodes_df[episodes_df['direction'] == -1.0])}")
    print(f"  Penny stock episodes: {len(episodes_df[episodes_df['is_penny_stock'] == True])}")

    print(f"\n  P&L Statistics:")
    print(f"    Long  - Mean: {episodes_df[episodes_df['direction'] == 1.0]['final_pnl_pct'].mean():.2%}, "
          f"Std: {episodes_df[episodes_df['direction'] == 1.0]['final_pnl_pct'].std():.2%}")
    print(f"    Short - Mean: {episodes_df[episodes_df['direction'] == -1.0]['final_pnl_pct'].mean():.2%}, "
          f"Std: {episodes_df[episodes_df['direction'] == -1.0]['final_pnl_pct'].std():.2%}")

    # Show extreme cases (potential catastrophic scenarios)
    worst_episodes = episodes_df.nsmallest(10, 'final_pnl_pct')
    print(f"\n  Worst 10 Episodes (catastrophic scenarios):")
    for _, ep in worst_episodes.iterrows():
        direction = "LONG" if ep['direction'] == 1.0 else "SHORT"
        print(f"    {ep['symbol']} ({direction}): {ep['final_pnl_pct']:.2%} - "
              f"{'PENNY' if ep['is_penny_stock'] else 'NORMAL'}")

print("\n" + "="*60)
print("FILES SAVED TO OBJECT STORE")
print("="*60)
print("\nTo download files:")
print("  1. Go to: https://www.quantconnect.com/terminal/#organization/object-store")
print("  2. Or use the Storage panel in QuantConnect")
print("  3. Download these keys:")
print(f"     - {CONFIG['output_episodes']}")
print(f"     - {CONFIG['output_prices']}")
print(f"     - {CONFIG['output_metadata']}")
print("\nAlternatively, read files in notebook with:")
print(f"  csv_data = qb.object_store.read('{CONFIG['output_episodes']}')")
print(f"  pkl_bytes = qb.object_store.read_bytes('{CONFIG['output_prices']}')")
print("="*60)
