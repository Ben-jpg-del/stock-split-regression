import os
import pickle
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# Rate limiting configuration
RATE_LIMIT_DELAY_MIN = 0.5  # Minimum delay between requests (seconds)
RATE_LIMIT_DELAY_MAX = 1.5  # Maximum delay between requests (seconds)
RATE_LIMIT_RETRY_ATTEMPTS = 3  # Number of retry attempts on failure
RATE_LIMIT_RETRY_DELAY = 5.0  # Delay between retries (seconds)
RATE_LIMIT_BATCH_SIZE = 10  # Download in batches to avoid rate limits
RATE_LIMIT_BATCH_DELAY = 3.0  # Delay between batches (seconds)


@dataclass
class TradeEpisode:
    symbol: str
    direction: float  # 1.0 long, -1.0 short
    entry_price: float
    entry_date: datetime
    price_trajectory: np.ndarray  # Hourly prices during the trade
    hold_duration: int  # Hours
    sector_roc: float  # Sector momentum at entry
    is_penny_stock: bool
    final_pnl_pct: float  # Actual outcome


class HistoricalDataLoader:
    def __init__(
        self,
        cache_dir: str = 'data/price_cache',
        tech_symbols: Optional[List[str]] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.tech_symbols = tech_symbols or [
            # === MEGA CAP TECH (>$500B) ===
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA',

            # === LARGE CAP SEMICONDUCTORS ===
            'AMD', 'INTC', 'AVGO', 'QCOM', 'TXN', 'MU', 'AMAT', 'LRCX',
            'KLAC', 'MRVL', 'NXPI', 'ADI', 'MCHP', 'ON', 'MPWR', 'SWKS',
            'QRVO', 'ENTG', 'TER', 'WOLF',

            # === LARGE CAP SOFTWARE ===
            'CRM', 'ORCL', 'ADBE', 'NOW', 'INTU', 'SNPS', 'CDNS', 'WDAY',
            'TEAM', 'ANSS', 'ADSK', 'PANW', 'FTNT', 'ZS', 'CRWD', 'OKTA',
            'DDOG', 'SNOW', 'NET', 'MDB', 'HUBS', 'SPLK', 'VEEV', 'PAYC',
            'PCTY', 'MANH', 'SMAR', 'FIVN', 'ZEN', 'NICE', 'DOCU', 'BILL',

            # === NETWORKING & COMMUNICATIONS ===
            'CSCO', 'ANET', 'JNPR', 'FFIV', 'NTAP', 'AKAM', 'CDN', 'ROKU',

            # === IT SERVICES & CONSULTING ===
            'ACN', 'IBM', 'CTSH', 'EPAM', 'GLOB', 'GDDY', 'TWLO', 'TTD',

            # === HARDWARE & EQUIPMENT ===
            'HPQ', 'HPE', 'DELL', 'WDC', 'STX', 'PSTG', 'LOGI', 'KEYS',

            # === MID CAP TECH ===
            'PLTR', 'RBLX', 'U', 'PATH', 'CFLT', 'DOCN', 'GTLB', 'ESTC',
            'PLAN', 'APPS', 'ZI', 'APPN', 'NCNO', 'ASAN', 'CWAN', 'BRZE',
            'FRSH', 'SUMO', 'FSLY', 'NEWR', 'DT', 'SPT', 'RPD', 'TOST',

            # === SMALL CAP / VOLATILE TECH (higher risk scenarios) ===
            'BIGC', 'AMPL', 'AVPT', 'BLZE', 'CXAI', 'DCBO', 'ENFN', 'FROG',
            'GENI', 'HLIT', 'IONQ', 'JAMF', 'KARO', 'KVYO', 'MTTR', 'OUST',
            'PRGS', 'QLYS', 'RCAT', 'SMCI', 'TDUP', 'UPST', 'VERX', 'VZIO',
            'WEAV', 'XRX', 'YEXT', 'ZETA',

            # === FINTECH (often classified as tech) ===
            'PYPL', 'SQ', 'AFRM', 'SOFI', 'COIN', 'HOOD', 'SHOP', 'MELI',

            # === CYBERSECURITY ===
            'S', 'CYBR', 'TENB', 'VRNS', 'QLYS', 'SAIL', 'TUFN',

            # === CHINESE TECH ADRs (higher volatility) ===
            'BABA', 'JD', 'PDD', 'BIDU', 'NTES', 'TME', 'BILI', 'IQ',
        ]

        # Sector ETF for momentum calculation
        self.sector_etf = 'XLK'

        # Loaded data cache
        self._price_data: Dict[str, pd.DataFrame] = {}
        self._sector_data: Optional[pd.DataFrame] = None

    def _download_with_retry(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        for attempt in range(RATE_LIMIT_RETRY_ATTEMPTS):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                )
                if len(df) > 0:
                    return df
                return None
            except Exception as e:
                error_msg = str(e).lower()
                # Check for rate limit indicators
                if 'rate' in error_msg or 'limit' in error_msg or '429' in error_msg:
                    wait_time = RATE_LIMIT_RETRY_DELAY * (attempt + 1)
                    print(f"    Rate limited, waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                elif attempt < RATE_LIMIT_RETRY_ATTEMPTS - 1:
                    print(f"    Error (attempt {attempt + 1}): {e}, retrying...")
                    time.sleep(RATE_LIMIT_RETRY_DELAY)
                else:
                    print(f"    Failed after {RATE_LIMIT_RETRY_ATTEMPTS} attempts: {e}")
                    return None
        return None

    def download_data(
        self,
        symbols: Optional[List[str]] = None,
        start_date: str = '2020-01-01',
        end_date: Optional[str] = None,
        interval: str = '1h',
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        symbols = symbols or self.tech_symbols
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')

        print(f"Downloading {len(symbols)} symbols from {start_date} to {end_date}...")
        print(f"Rate limiting: {RATE_LIMIT_DELAY_MIN}-{RATE_LIMIT_DELAY_MAX}s delay, "
              f"batch size {RATE_LIMIT_BATCH_SIZE}")

        data = {}
        symbols_to_download = []
        cached_count = 0

        # First pass: check cache and collect symbols to download
        for symbol in symbols:
            cache_file = self.cache_dir / f"{symbol}_{interval}_{start_date}_{end_date}.pkl"

            if cache_file.exists() and not force_refresh:
                try:
                    with open(cache_file, 'rb') as f:
                        data[symbol] = pickle.load(f)
                    cached_count += 1
                except Exception:
                    symbols_to_download.append(symbol)
            else:
                symbols_to_download.append(symbol)

        if cached_count > 0:
            print(f"  Loaded {cached_count} symbols from cache")

        # Second pass: download in batches with rate limiting
        if symbols_to_download:
            print(f"  Downloading {len(symbols_to_download)} symbols...")

            for batch_idx in range(0, len(symbols_to_download), RATE_LIMIT_BATCH_SIZE):
                batch = symbols_to_download[batch_idx:batch_idx + RATE_LIMIT_BATCH_SIZE]
                batch_num = batch_idx // RATE_LIMIT_BATCH_SIZE + 1
                total_batches = (len(symbols_to_download) + RATE_LIMIT_BATCH_SIZE - 1) // RATE_LIMIT_BATCH_SIZE

                print(f"  Batch {batch_num}/{total_batches}: {len(batch)} symbols")

                for i, symbol in enumerate(batch):
                    progress = batch_idx + i + 1
                    print(f"    [{progress}/{len(symbols_to_download)}] Downloading {symbol}...", end='')

                    df = self._download_with_retry(symbol, start_date, end_date, interval)

                    if df is not None:
                        data[symbol] = df
                        # Cache for later
                        cache_file = self.cache_dir / f"{symbol}_{interval}_{start_date}_{end_date}.pkl"
                        try:
                            with open(cache_file, 'wb') as f:
                                pickle.dump(df, f)
                            print(f" OK ({len(df)} rows)")
                        except Exception as e:
                            print(f" OK (cache failed: {e})")
                    else:
                        print(" FAILED")

                    # Rate limit delay between requests
                    if i < len(batch) - 1:  # Don't delay after last in batch
                        delay = random.uniform(RATE_LIMIT_DELAY_MIN, RATE_LIMIT_DELAY_MAX)
                        time.sleep(delay)

                # Delay between batches
                if batch_idx + RATE_LIMIT_BATCH_SIZE < len(symbols_to_download):
                    print(f"    Batch complete, waiting {RATE_LIMIT_BATCH_DELAY}s before next batch...")
                    time.sleep(RATE_LIMIT_BATCH_DELAY)

        # Also download sector ETF
        print(f"  Downloading sector ETF ({self.sector_etf})...")
        sector_df = self._download_with_retry(
            self.sector_etf, start_date, end_date, '1d'
        )
        if sector_df is not None:
            self._sector_data = sector_df
            print(f"    Sector ETF: {len(sector_df)} rows")
        else:
            print(f"    Warning: Failed to download sector ETF {self.sector_etf}")

        self._price_data = data
        print(f"Download complete: {len(data)} symbols successfully loaded")
        return data

    def calculate_sector_roc(self, date: datetime, lookback: int = 22) -> float:
        if self._sector_data is None or len(self._sector_data) == 0:
            return 0.0

        # Find closest date in sector data
        mask = self._sector_data.index <= date
        if not mask.any():
            return 0.0

        current_idx = mask.sum() - 1
        if current_idx < lookback:
            return 0.0

        current_price = self._sector_data['Close'].iloc[current_idx]
        past_price = self._sector_data['Close'].iloc[current_idx - lookback]

        if past_price > 0:
            return (current_price - past_price) / past_price
        return 0.0

    def create_training_episodes(
        self,
        symbols: Optional[List[str]] = None,
        min_episode_length: int = 24,  # Minimum 24 hours (1 day)
        max_episode_length: int = 72,  # Maximum 72 hours (3 days)
        episodes_per_symbol: int = 100,
        include_shorts: bool = True,
        penny_stock_threshold: float = 5.0,  # Consider < $5 as small cap
    ) -> List[TradeEpisode]:
        symbols = symbols or list(self._price_data.keys())
        episodes = []

        print(f"Creating training episodes from {len(symbols)} symbols...")

        for symbol in symbols:
            if symbol not in self._price_data:
                continue

            df = self._price_data[symbol]
            if len(df) < max_episode_length:
                continue

            # Create random episodes
            for _ in range(episodes_per_symbol):
                # Random start point (leaving room for episode)
                max_start = len(df) - max_episode_length
                if max_start <= 0:
                    continue

                start_idx = np.random.randint(0, max_start)

                # Random episode length
                ep_length = np.random.randint(min_episode_length, max_episode_length + 1)

                # Extract price trajectory
                prices = df['Close'].iloc[start_idx:start_idx + ep_length].values

                if len(prices) < min_episode_length or np.isnan(prices).any():
                    continue

                entry_price = prices[0]
                entry_date = df.index[start_idx]

                # Random direction (or always long if shorts disabled)
                if include_shorts:
                    direction = np.random.choice([1.0, -1.0])
                else:
                    direction = 1.0

                # Calculate actual P&L
                exit_price = prices[-1]
                pnl_pct = direction * (exit_price - entry_price) / entry_price

                # Get sector momentum
                sector_roc = self.calculate_sector_roc(entry_date)

                # Check if penny/small cap stock
                is_penny = entry_price < penny_stock_threshold

                episodes.append(TradeEpisode(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    entry_date=entry_date,
                    price_trajectory=prices,
                    hold_duration=ep_length,
                    sector_roc=sector_roc,
                    is_penny_stock=is_penny,
                    final_pnl_pct=pnl_pct,
                ))

        print(f"Created {len(episodes)} training episodes")

        if not episodes:
            print("  No episodes created - check data availability")
            return episodes

        # Print statistics
        pnls = [e.final_pnl_pct for e in episodes]
        penny_count = sum(1 for e in episodes if e.is_penny_stock)
        print(f"  P&L range: [{min(pnls):.2%}, {max(pnls):.2%}]")
        print(f"  Mean P&L: {np.mean(pnls):.2%}")
        print(f"  Penny stocks: {penny_count} ({penny_count/len(episodes):.1%})")

        return episodes

    def create_catastrophic_scenarios(
        self,
        num_scenarios: int = 100,
        max_adverse_move: float = 5.0,  # Up to 500% adverse move
    ) -> List[TradeEpisode]:
        episodes = []
        print(f"Creating {num_scenarios} catastrophic scenarios...")

        for i in range(num_scenarios):
            # Random parameters
            direction = np.random.choice([1.0, -1.0])
            entry_price = np.random.uniform(0.1, 10.0)  # Often penny stocks
            ep_length = np.random.randint(24, 72)

            # Create adverse price trajectory
            # Gradually accelerates against the position
            t = np.linspace(0, 1, ep_length)

            # Exponential adverse move
            adverse_mult = 1 + (max_adverse_move - 1) * np.random.random()

            if direction == 1.0:  # Long position, price crashes
                multiplier = 1 - (1 - 1/adverse_mult) * (t ** 2)
            else:  # Short position, price explodes (like TGL)
                multiplier = 1 + (adverse_mult - 1) * (t ** 2)

            prices = entry_price * multiplier

            # Add some noise
            noise = np.random.normal(0, 0.02, ep_length)
            prices = prices * (1 + noise)
            prices = np.maximum(prices, 0.01)  # Floor at 1 cent

            exit_price = prices[-1]
            pnl_pct = direction * (exit_price - entry_price) / entry_price

            episodes.append(TradeEpisode(
                symbol=f"CATASTROPHIC_{i}",
                direction=direction,
                entry_price=entry_price,
                entry_date=datetime.now() - timedelta(days=np.random.randint(0, 365)),
                price_trajectory=prices,
                hold_duration=ep_length,
                sector_roc=np.random.uniform(-0.1, 0.1),
                is_penny_stock=entry_price < 5.0,
                final_pnl_pct=pnl_pct,
            ))

        print(f"Created {len(episodes)} catastrophic scenarios")
        pnls = [e.final_pnl_pct for e in episodes]
        print(f"  P&L range: [{min(pnls):.2%}, {max(pnls):.2%}]")

        return episodes

    def load_quantconnect_trades(
        self,
        trades_csv: str,
        logs_txt: Optional[str] = None,
    ) -> List[TradeEpisode]:
        print(f"Loading trades from {trades_csv}...")

        try:
            trades_df = pd.read_csv(trades_csv)
        except Exception as e:
            print(f"Error loading trades: {e}")
            return []

        episodes = []
        # Group trades by symbol and process
        # This is simplified - would need actual price data to build trajectories

        for _, row in trades_df.iterrows():
            try:
                symbol = row.get('Symbol', 'UNKNOWN')
                quantity = row.get('Quantity', 0)
                fill_price = row.get('Fill Price', 0)

                if quantity == 0 or fill_price == 0:
                    continue

                direction = 1.0 if quantity > 0 else -1.0
                is_penny = fill_price < 5.0

                # Create a simple episode (would need actual price trajectory)
                episodes.append(TradeEpisode(
                    symbol=symbol,
                    direction=direction,
                    entry_price=fill_price,
                    entry_date=datetime.now(),  # Would parse from row
                    price_trajectory=np.array([fill_price]),  # Placeholder
                    hold_duration=72,  # 3 days
                    sector_roc=0.0,
                    is_penny_stock=is_penny,
                    final_pnl_pct=0.0,  # Would calculate from exit
                ))
            except Exception as e:
                continue

        print(f"Loaded {len(episodes)} trades from backtest")
        return episodes

    def get_episode_as_env_data(
        self,
        episode: TradeEpisode,
    ) -> Dict:
        return {
            'symbol': episode.symbol,
            'direction': episode.direction,
            'entry_price': episode.entry_price,
            'prices': episode.price_trajectory,
            'sector_roc': episode.sector_roc,
            'is_penny_stock': episode.is_penny_stock,
            'hold_duration': episode.hold_duration,
        }


class HistoricalEpisodeEnv:
    def __init__(
        self,
        episodes: List[TradeEpisode],
        reward_config=None,
    ):
        from .reward_functions import SACRiskReward, RiskRewardConfig
        from .environment import TradeState

        self.episodes = episodes
        self.reward_fn = SACRiskReward(reward_config)

        self.state_dim = 12
        self.action_dim = 1

        self._current_episode: Optional[TradeEpisode] = None
        self._current_step = 0
        self._position_multiplier = 1.0
        self._trade_state = None

    def reset(self, episode_idx: Optional[int] = None) -> np.ndarray:
        if episode_idx is None:
            episode_idx = np.random.randint(len(self.episodes))

        self._current_episode = self.episodes[episode_idx]
        self._current_step = 0
        self._position_multiplier = 1.0
        self.reward_fn.reset()

        # Initialize trade state from episode
        ep = self._current_episode
        from .environment import TradeState

        self._trade_state = TradeState(
            direction=ep.direction,
            size_pct=0.25,  # Default position size
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
        state.days_held = self._current_step // 24  # Hours to days

        # Update margin (simplified)
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

        # Check done
        done = info['margin_call'] or info['trade_closed']

        return self._trade_state.to_array(), reward, done, info


def download_and_prepare_data(
    start_date: str = '2020-01-01',
    end_date: Optional[str] = None,
    cache_dir: str = 'data/price_cache',
) -> Tuple[List[TradeEpisode], List[TradeEpisode]]:
    """
    Convenience function to download data and create training episodes.

    Returns:
        Tuple of (normal_episodes, catastrophic_episodes)
    """
    loader = HistoricalDataLoader(cache_dir=cache_dir)

    # Download data
    loader.download_data(
        start_date=start_date,
        end_date=end_date,
        interval='1h',
    )

    # Create normal episodes
    normal_episodes = loader.create_training_episodes(
        episodes_per_symbol=100,
    )

    # Create catastrophic scenarios
    catastrophic_episodes = loader.create_catastrophic_scenarios(
        num_scenarios=200,
    )

    return normal_episodes, catastrophic_episodes


if __name__ == "__main__":
    # Test the data loader
    print("Testing HistoricalDataLoader...")

    loader = HistoricalDataLoader()

    # Test with a few symbols
    test_symbols = ['AAPL', 'MSFT', 'NVDA']

    if YFINANCE_AVAILABLE:
        print("\nDownloading test data...")
        data = loader.download_data(
            symbols=test_symbols,
            start_date='2024-01-01',
            interval='1h',
        )

        print("\nCreating training episodes...")
        episodes = loader.create_training_episodes(
            symbols=test_symbols,
            episodes_per_symbol=10,
        )

        if episodes:
            print(f"\nSample episode:")
            ep = episodes[0]
            print(f"  Symbol: {ep.symbol}")
            print(f"  Direction: {'LONG' if ep.direction > 0 else 'SHORT'}")
            print(f"  Entry price: ${ep.entry_price:.2f}")
            print(f"  Duration: {ep.hold_duration} hours")
            print(f"  Final P&L: {ep.final_pnl_pct:.2%}")
    else:
        print("\nyfinance not installed. Install with: pip install yfinance")

    # Test catastrophic scenarios
    print("\nCreating catastrophic scenarios...")
    catastrophic = loader.create_catastrophic_scenarios(num_scenarios=10)

    print(f"\nSample catastrophic episode:")
    ep = catastrophic[0]
    print(f"  Direction: {'LONG' if ep.direction > 0 else 'SHORT'}")
    print(f"  Entry price: ${ep.entry_price:.2f}")
    print(f"  Exit price: ${ep.price_trajectory[-1]:.2f}")
    print(f"  Final P&L: {ep.final_pnl_pct:.2%}")

    print("\n Data loader working correctly!")
