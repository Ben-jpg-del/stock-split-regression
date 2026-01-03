"""
QuantConnect Data Loader for SAC Training

This module loads training data exported from QuantConnect's QuantBook.
Use this instead of yfinance for higher quality, longer historical data.

Usage:
    from RL.risk_management.qc_data_loader import load_qc_episodes

    train_episodes, eval_episodes = load_qc_episodes(
        episodes_file='research/data/price_trajectories.pkl',
        train_ratio=0.8
    )
"""

import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class QCTradeEpisode:
    """Trade episode loaded from QuantConnect export."""
    symbol: str
    direction: float  # 1.0 long, -1.0 short
    entry_price: float
    exit_price: float
    entry_date: datetime
    price_trajectory: np.ndarray
    hold_duration: int  # Hours
    sector_roc: float
    is_penny_stock: bool
    final_pnl_pct: float
    split_factor: float


def load_qc_episodes(
    episodes_file: str = 'research/data/price_trajectories.pkl',
    metadata_file: Optional[str] = None,
    train_ratio: float = 0.8,
    include_catastrophic_synthetic: bool = False,
    catastrophic_ratio: float = 0.5,
    min_trajectory_length: int = 24,
) -> Tuple[List[QCTradeEpisode], List[QCTradeEpisode]]:
    """
    Load episodes exported from QuantConnect and split into train/eval.

    Args:
        episodes_file: Path to pickle file with price trajectories
        metadata_file: Optional path to metadata JSON
        train_ratio: Ratio of data for training (oldest data)
        include_catastrophic_synthetic: Add synthetic catastrophic scenarios
        catastrophic_ratio: Ratio of catastrophic to normal episodes
        min_trajectory_length: Minimum price trajectory length in hours

    Returns:
        Tuple of (train_episodes, eval_episodes)
    """
    episodes_path = Path(episodes_file)

    if not episodes_path.exists():
        raise FileNotFoundError(
            f"Episodes file not found: {episodes_file}\n"
            "Please run the QuantConnect export script first:\n"
            "  1. Go to QuantConnect Research\n"
            "  2. Run research/export_training_data.py\n"
            "  3. Download price_trajectories.pkl"
        )

    print(f"Loading QuantConnect episodes from {episodes_file}...")

    # Load episodes
    with open(episodes_path, 'rb') as f:
        raw_episodes = pickle.load(f)

    print(f"  Raw episodes loaded: {len(raw_episodes)}")

    # Convert to QCTradeEpisode objects with data validation
    episodes = []
    skipped_short = 0
    skipped_price = 0
    skipped_pnl = 0

    # Data quality thresholds
    MAX_VALID_PRICE = 100000  # $100k max (catches corrupted data like $2 quintillion)
    MAX_PNL_MAGNITUDE = 5.0   # 500% max P&L (clip extreme outliers)

    for ep in raw_episodes:
        # Parse entry date
        try:
            entry_date = datetime.fromisoformat(ep['split_time'])
        except (KeyError, ValueError):
            entry_date = datetime.now()

        # Get price trajectory
        trajectory = np.array(ep['price_trajectory'])

        # Skip if trajectory too short
        if len(trajectory) < min_trajectory_length:
            skipped_short += 1
            continue

        # Skip corrupted price data (catches overflow issues)
        entry_price = ep['entry_price']
        if entry_price > MAX_VALID_PRICE or entry_price <= 0:
            skipped_price += 1
            continue

        # Skip episodes with NaN/Inf P&L
        final_pnl = ep['final_pnl_pct']
        if np.isnan(final_pnl) or np.isinf(final_pnl):
            skipped_pnl += 1
            continue

        # Clip extreme P&L values (but don't skip - these are training signals)
        final_pnl = np.clip(final_pnl, -MAX_PNL_MAGNITUDE, MAX_PNL_MAGNITUDE)

        episodes.append(QCTradeEpisode(
            symbol=ep['symbol'],
            direction=ep['direction'],
            entry_price=entry_price,
            exit_price=ep['exit_price'],
            entry_date=entry_date,
            price_trajectory=trajectory,
            hold_duration=ep.get('hold_duration_hours', len(trajectory)),
            sector_roc=ep['sector_roc'],
            is_penny_stock=ep['is_penny_stock'],
            final_pnl_pct=final_pnl,
            split_factor=ep.get('split_factor', 1.0),
        ))

    total_skipped = skipped_short + skipped_price + skipped_pnl
    print(f"  Valid episodes: {len(episodes)} (skipped {total_skipped})")
    if skipped_price > 0:
        print(f"    - Skipped {skipped_price} with corrupted prices (>${MAX_VALID_PRICE:,})")
    if skipped_short > 0:
        print(f"    - Skipped {skipped_short} with short trajectories (<{min_trajectory_length})")
    if skipped_pnl > 0:
        print(f"    - Skipped {skipped_pnl} with NaN/Inf P&L")

    # Load metadata if available
    if metadata_file:
        metadata_path = Path(metadata_file)
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"  Export date: {metadata.get('export_date', 'unknown')}")
            print(f"  Date range: {metadata.get('date_range', {})}")

    # Group episodes by their original split event (symbol + date)
    # This prevents data leakage from augmented episodes of the same split
    from collections import defaultdict
    split_events = defaultdict(list)
    for ep in episodes:
        event_key = (ep.symbol, ep.entry_date.date())
        split_events[event_key].append(ep)

    # Sort split events by date
    sorted_events = sorted(split_events.items(), key=lambda x: x[0][1])

    # Split at the event level, not episode level
    split_idx = int(len(sorted_events) * train_ratio)
    train_events = sorted_events[:split_idx]
    eval_events = sorted_events[split_idx:]

    # Flatten back to episode lists
    train_episodes = [ep for _, eps in train_events for ep in eps]
    eval_episodes = [ep for _, eps in eval_events for ep in eps]

    print(f"\n  TIME-BASED SPLIT (by split event to prevent leakage):")
    print(f"    Unique split events: {len(sorted_events)}")
    print(f"    Training: {len(train_episodes)} episodes from {len(train_events)} events")
    print(f"    Evaluation: {len(eval_episodes)} episodes from {len(eval_events)} events")

    if train_events and eval_events:
        train_start = train_events[0][0][1]
        train_end = train_events[-1][0][1]
        eval_start = eval_events[0][0][1]
        eval_end = eval_events[-1][0][1]
        print(f"    Training range: {train_start} to {train_end}")
        print(f"    Eval range: {eval_start} to {eval_end}")
        print(f"    No overlap: {train_end < eval_start}")

    # Add synthetic catastrophic scenarios to training only
    if include_catastrophic_synthetic:
        num_catastrophic = int(len(train_episodes) * catastrophic_ratio)
        catastrophic = create_catastrophic_scenarios(num_catastrophic)
        train_episodes = train_episodes + catastrophic
        print(f"\n  Added {num_catastrophic} synthetic catastrophic scenarios to training")

    # Print statistics
    _print_episode_stats(train_episodes, "Training")
    _print_episode_stats(eval_episodes, "Evaluation")

    return train_episodes, eval_episodes


def create_catastrophic_scenarios(
    num_scenarios: int = 200,  # Increased default
    max_adverse_move: float = 10.0,  # Increased from 5.0 to 10.0 (up to 1000% adverse move)
) -> List[QCTradeEpisode]:
    """
    Create synthetic catastrophic scenarios for robust training.

    These simulate extreme events like the TGL disaster where
    a short position faces a massive adverse price move.
    """
    from datetime import timedelta

    episodes = []

    for i in range(num_scenarios):
        # Random parameters
        direction = np.random.choice([1.0, -1.0])
        entry_price = np.random.uniform(0.1, 10.0)  # Often penny stocks
        ep_length = np.random.randint(24, 72)

        # Create adverse price trajectory
        t = np.linspace(0, 1, ep_length)

        # Exponential adverse move
        adverse_mult = 1 + (max_adverse_move - 1) * np.random.random()

        if direction == 1.0:  # Long position, price crashes
            multiplier = 1 - (1 - 1/adverse_mult) * (t ** 2)
        else:  # Short position, price explodes (like TGL)
            multiplier = 1 + (adverse_mult - 1) * (t ** 2)

        prices = entry_price * multiplier

        # Add noise
        noise = np.random.normal(0, 0.02, ep_length)
        prices = prices * (1 + noise)
        prices = np.maximum(prices, 0.01)

        exit_price = prices[-1]
        pnl_pct = direction * (exit_price - entry_price) / entry_price

        episodes.append(QCTradeEpisode(
            symbol=f"CATASTROPHIC_{i}",
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_date=datetime.now() - timedelta(days=np.random.randint(0, 365)),
            price_trajectory=prices,
            hold_duration=ep_length,
            sector_roc=np.random.uniform(-0.1, 0.1),
            is_penny_stock=entry_price < 5.0,
            final_pnl_pct=pnl_pct,
            split_factor=1.0,
        ))

    return episodes


def _print_episode_stats(episodes: List[QCTradeEpisode], name: str):
    """Print statistics for a set of episodes."""
    if not episodes:
        print(f"\n  {name}: No episodes")
        return

    pnls = [e.final_pnl_pct for e in episodes]
    penny_count = sum(1 for e in episodes if e.is_penny_stock)
    long_count = sum(1 for e in episodes if e.direction > 0)
    short_count = sum(1 for e in episodes if e.direction < 0)

    print(f"\n  {name} Statistics:")
    print(f"    Episodes: {len(episodes)}")
    print(f"    Long/Short: {long_count}/{short_count}")
    print(f"    Penny stocks: {penny_count} ({penny_count/len(episodes):.1%})")
    print(f"    P&L - Mean: {np.mean(pnls):.2%}, Std: {np.std(pnls):.2%}")
    print(f"    P&L - Min: {np.min(pnls):.2%}, Max: {np.max(pnls):.2%}")


class QCHistoricalEpisodeEnv:
    """
    Environment wrapper for QuantConnect exported episodes.

    Compatible with the existing training pipeline.
    """

    def __init__(
        self,
        episodes: List[QCTradeEpisode],
        reward_config=None,
        reward_type: str = 'simple',
    ):
        from .reward_functions import create_reward_function
        from .environment import TradeState

        self.episodes = episodes
        self.reward_fn = create_reward_function(reward_type, reward_config)

        self.state_dim = 12
        self.action_dim = 1

        self._current_episode: Optional[QCTradeEpisode] = None
        self._current_step = 0
        self._position_multiplier = 1.0
        self._trade_state = None

    def reset(self, episode_idx: Optional[int] = None) -> np.ndarray:
        """Reset to a new episode."""
        if episode_idx is None:
            episode_idx = np.random.randint(len(self.episodes))

        self._current_episode = self.episodes[episode_idx]
        self._current_step = 0
        self._position_multiplier = 1.0
        self.reward_fn.reset()

        ep = self._current_episode
        from .environment import TradeState

        self._trade_state = TradeState(
            direction=ep.direction,
            size_pct=0.25,
            days_held=0,
            predicted_return=ep.final_pnl_pct * 0.5,  # Noisy prediction
            entry_price=ep.entry_price,
            current_price=ep.entry_price,
            unrealized_pnl_pct=0.0,
            max_drawdown_pct=0.0,
            peak_pnl_pct=0.0,
            pnl_velocity=0.0,
            sector_roc=ep.sector_roc,
            volatility_ratio=1.0,
            price_vs_sma=0.0,
            margin_usage=0.5,
            portfolio_heat=0.4,
            is_penny_stock=ep.is_penny_stock,
            stock_price=ep.entry_price,
        )

        return self._trade_state.to_array()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step using historical price data."""
        assert self._current_episode is not None, "Call reset() first"

        ep = self._current_episode
        state = self._trade_state

        # Store previous values
        prev_pnl = state.unrealized_pnl_pct
        prev_position = self._position_multiplier

        # Apply action
        self._position_multiplier = (action + 1) / 2

        # Advance using historical prices
        self._current_step += 1
        if self._current_step < len(ep.price_trajectory):
            new_price = ep.price_trajectory[self._current_step]
        else:
            new_price = ep.price_trajectory[-1]

        # Update state
        price_change = (new_price - state.current_price) / state.current_price
        state.current_price = new_price
        state.is_penny_stock = new_price < 5.0
        state.stock_price = new_price

        # Update P&L
        base_pnl_change = price_change * state.direction
        scaled_pnl_change = base_pnl_change * self._position_multiplier
        state.unrealized_pnl_pct += scaled_pnl_change

        # Update peak/drawdown
        if state.unrealized_pnl_pct > state.peak_pnl_pct:
            state.peak_pnl_pct = state.unrealized_pnl_pct
        current_dd = state.peak_pnl_pct - state.unrealized_pnl_pct
        if current_dd > state.max_drawdown_pct:
            state.max_drawdown_pct = current_dd

        state.pnl_velocity = state.unrealized_pnl_pct - prev_pnl
        state.size_pct *= self._position_multiplier
        state.days_held = self._current_step // 24

        # Update margin
        if state.unrealized_pnl_pct < 0:
            state.margin_usage = min(1.0, 0.5 - state.unrealized_pnl_pct * 2)
        else:
            state.margin_usage = max(0.1, 0.5 - state.unrealized_pnl_pct * 0.5)

        # Build info dict
        info = {
            'margin_call': state.margin_usage > 0.95,
            'pnl_change': state.unrealized_pnl_pct - prev_pnl,
            'wealth': 1.0 + state.unrealized_pnl_pct,
            'max_drawdown_pct': state.max_drawdown_pct,
            'new_position': self._position_multiplier,
            'old_position': prev_position,
            'is_penny_stock': state.is_penny_stock,
            'position_pct': state.size_pct,
            'volatility_ratio': state.volatility_ratio,
            'pnl_velocity': state.pnl_velocity,
            'trade_closed': self._current_step >= len(ep.price_trajectory) - 1,
            'realized_pnl_pct': state.unrealized_pnl_pct if self._current_step >= len(ep.price_trajectory) - 1 else 0.0,
            'days_held': state.days_held,
        }

        # Calculate reward
        reward = self.reward_fn.calculate_reward(
            self._trade_state.to_array(),
            action,
            self._trade_state.to_array(),
            info
        )

        done = info['margin_call'] or info['trade_closed']

        return self._trade_state.to_array(), reward, done, info


if __name__ == "__main__":
    # Test loading (will fail if no data exported yet)
    print("Testing QC Data Loader...")

    try:
        train_eps, eval_eps = load_qc_episodes(
            episodes_file='research/data/price_trajectories.pkl',
            train_ratio=0.8
        )
        print(f"\nLoaded {len(train_eps)} training, {len(eval_eps)} eval episodes")

        # Test environment
        if train_eps:
            env = QCHistoricalEpisodeEnv(train_eps)
            state = env.reset()
            print(f"State shape: {state.shape}")

            action = np.array([0.5])
            next_state, reward, done, info = env.step(action[0])
            print(f"Step: reward={reward:.4f}, done={done}")

    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nTo generate data:")
        print("  1. Open QuantConnect Research")
        print("  2. Run research/export_training_data.py")
        print("  3. Download the generated files")
