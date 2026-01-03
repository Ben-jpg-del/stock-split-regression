import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from .reward_functions import SACRiskReward, RiskRewardConfig


@dataclass
class TradeState:
    direction: float = 0.0         # 1.0 long, -1.0 short, 0.0 flat
    size_pct: float = 0.0          # Position size as % of portfolio
    days_held: int = 0             # Days in trade
    predicted_return: float = 0.0  # Model's predicted return

    # P&L info
    entry_price: float = 0.0       # Entry price
    current_price: float = 0.0     # Current price
    unrealized_pnl_pct: float = 0.0  # Unrealized P&L as %
    max_drawdown_pct: float = 0.0  # Max drawdown during trade
    peak_pnl_pct: float = 0.0      # Peak unrealized P&L
    pnl_velocity: float = 0.0      # Rate of P&L change

    # Market conditions
    sector_roc: float = 0.0        # Sector momentum (XLK ROC)
    volatility_ratio: float = 1.0  # Current vol / historical vol
    price_vs_sma: float = 0.0      # Price relative to 20-day SMA

    # Risk metrics
    margin_usage: float = 0.0      # Current margin utilization
    portfolio_heat: float = 0.0    # Total portfolio risk exposure

    # Stock info
    is_penny_stock: bool = False   # True if price < $1
    stock_price: float = 0.0       # Current stock price

    def to_array(self) -> np.ndarray:
        """Convert state to 12-dimensional numpy array."""
        return np.array([
            self.direction,
            self.size_pct,
            self.days_held / 3.0,  # Normalize by hold duration
            self.predicted_return,
            self.unrealized_pnl_pct,
            self.max_drawdown_pct,
            self.pnl_velocity,
            self.sector_roc,
            self.volatility_ratio,
            self.price_vs_sma,
            self.margin_usage,
            self.portfolio_heat,
        ], dtype=np.float32)


class TradingRiskEnv:
    """
    Gym-compatible environment for SAC risk management training.

    The environment simulates individual trade episodes where the agent
    must decide how much position to maintain (via position multiplier).

    State Space (12 dimensions):
        0. direction: Trade direction (1=long, -1=short, 0=flat)
        1. size_pct: Position size as % of portfolio
        2. days_held: Normalized days in trade (0-1)
        3. predicted_return: Model's predicted return
        4. unrealized_pnl_pct: Current unrealized P&L %
        5. max_drawdown_pct: Maximum drawdown during trade
        6. pnl_velocity: Rate of P&L change
        7. sector_roc: Sector momentum indicator
        8. volatility_ratio: Current vol / historical vol
        9. price_vs_sma: Price relative to moving average
        10. margin_usage: Current margin utilization
        11. portfolio_heat: Total portfolio risk exposure

    Action Space:
        Continuous [-1, 1] mapped to position multiplier [0, 1]
        -1 = exit completely
         0 = 50% position
        +1 = full position

    Reward:
        From SACRiskReward function (asymmetric, risk-adjusted)
    """

    def __init__(
        self,
        historical_trades: Optional[pd.DataFrame] = None,
        reward_config: Optional[RiskRewardConfig] = None,
        max_steps: int = 10,
        hold_duration: int = 3,
        initial_margin_usage: float = 0.5,
        synthetic_mode: bool = True,
    ):
        """
        Initialize the trading environment.

        Args:
            historical_trades: DataFrame with historical trade data
            reward_config: Reward function configuration
            max_steps: Maximum steps per episode
            hold_duration: Expected hold duration in days
            initial_margin_usage: Starting margin usage
            synthetic_mode: If True, generate synthetic trades
        """
        self.historical_trades = historical_trades
        self.max_steps = max_steps
        self.hold_duration = hold_duration
        self.initial_margin_usage = initial_margin_usage
        self.synthetic_mode = synthetic_mode

        # Dimensions
        self.state_dim = 12
        self.action_dim = 1

        # Reward function
        self.reward_fn = SACRiskReward(reward_config)

        # Episode state
        self._trade_state: Optional[TradeState] = None
        self._current_step = 0
        self._position_multiplier = 1.0
        self._episode_trades = []

        # For synthetic data generation
        self._price_history: List[float] = []

    def reset(self, trade_idx: Optional[int] = None) -> np.ndarray:
        """
        Reset environment for a new episode.

        Args:
            trade_idx: Optional index into historical trades

        Returns:
            Initial state observation (12-dim numpy array)
        """
        self._current_step = 0
        self._position_multiplier = 1.0
        self.reward_fn.reset()
        self._price_history = []

        if self.synthetic_mode or self.historical_trades is None:
            self._trade_state = self._generate_synthetic_trade()
        else:
            if trade_idx is None:
                trade_idx = np.random.randint(len(self.historical_trades))
            self._trade_state = self._load_historical_trade(trade_idx)

        return self._trade_state.to_array()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Continuous action in [-1, 1]

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        assert self._trade_state is not None, "Call reset() first"

        # Store current state for reward calculation
        prev_state = self._trade_state.to_array()
        prev_pnl = self._trade_state.unrealized_pnl_pct
        prev_position = self._position_multiplier

        # Apply action (update position multiplier)
        self._position_multiplier = (action + 1) / 2  # Map [-1,1] to [0,1]

        # Simulate market step
        self._simulate_step()

        # Get new state
        next_state = self._trade_state.to_array()

        # Build info dict
        info = self._get_step_info(prev_pnl, prev_position)

        # Calculate reward
        reward = self.reward_fn.calculate_reward(
            prev_state, action, next_state, info
        )

        # Check termination conditions
        done = self._is_done(info)

        # Update step counter
        self._current_step += 1

        return next_state, reward, done, info

    def _simulate_step(self):
        """Simulate one step of market movement."""
        state = self._trade_state

        # Store previous P&L for velocity calculation
        prev_pnl = state.unrealized_pnl_pct

        # Simulate price movement
        if self.synthetic_mode:
            price_change = self._generate_price_change()
        else:
            price_change = self._get_historical_price_change()

        # Update current price
        state.current_price *= (1 + price_change)
        self._price_history.append(state.current_price)

        # Check if penny stock
        state.is_penny_stock = state.current_price < 1.0
        state.stock_price = state.current_price

        # Update P&L (scaled by position multiplier)
        base_pnl_change = price_change * state.direction
        scaled_pnl_change = base_pnl_change * self._position_multiplier

        state.unrealized_pnl_pct += scaled_pnl_change

        # Update peak and drawdown
        if state.unrealized_pnl_pct > state.peak_pnl_pct:
            state.peak_pnl_pct = state.unrealized_pnl_pct

        current_dd = state.peak_pnl_pct - state.unrealized_pnl_pct
        if current_dd > state.max_drawdown_pct:
            state.max_drawdown_pct = current_dd

        # Update P&L velocity
        state.pnl_velocity = state.unrealized_pnl_pct - prev_pnl

        # Update position size (affected by multiplier)
        state.size_pct *= self._position_multiplier

        # Update days held
        state.days_held += 1

        # Update market conditions (simulate some noise)
        state.sector_roc += np.random.normal(0, 0.01)
        state.volatility_ratio = max(0.5, min(3.0,
            state.volatility_ratio + np.random.normal(0, 0.1)
        ))

        # Update margin usage (increases with losses)
        if state.unrealized_pnl_pct < 0:
            state.margin_usage = min(1.0,
                self.initial_margin_usage - state.unrealized_pnl_pct * 2
            )
        else:
            state.margin_usage = max(0.1,
                self.initial_margin_usage - state.unrealized_pnl_pct * 0.5
            )

    def _generate_price_change(self) -> float:
        """Generate synthetic price change with fat tails."""
        # Base volatility
        base_vol = 0.02

        # Single random draw for mutually exclusive outcomes
        r = np.random.random()
        if r < 0.05:
            # 5% chance of large move
            return np.random.normal(0, base_vol * 5)
        elif r < 0.06:
            # 1% chance of extreme move (like TGL)
            return np.random.choice([-1, 1]) * np.random.uniform(0.5, 2.0)

        return np.random.normal(0, base_vol)

    def _get_historical_price_change(self) -> float:
        """Get price change from historical data."""
        # This would be implemented with actual historical data
        return np.random.normal(0, 0.02)

    def _generate_synthetic_trade(self) -> TradeState:
        """Generate a synthetic trade for training."""
        # Random direction
        direction = np.random.choice([-1.0, 1.0])

        # Random initial conditions
        entry_price = np.random.uniform(1.0, 500.0)

        # Penny stocks are rare but important
        if np.random.random() < 0.1:
            entry_price = np.random.uniform(0.1, 1.0)

        return TradeState(
            direction=direction,
            size_pct=np.random.uniform(0.1, 0.3),
            days_held=0,
            predicted_return=np.random.normal(0, 0.05) * direction,
            entry_price=entry_price,
            current_price=entry_price,
            unrealized_pnl_pct=0.0,
            max_drawdown_pct=0.0,
            peak_pnl_pct=0.0,
            pnl_velocity=0.0,
            sector_roc=np.random.normal(0, 0.05),
            volatility_ratio=np.random.uniform(0.8, 1.5),
            price_vs_sma=np.random.normal(0, 0.05),
            margin_usage=self.initial_margin_usage,
            portfolio_heat=np.random.uniform(0.2, 0.6),
            is_penny_stock=entry_price < 1.0,
            stock_price=entry_price,
        )

    def _load_historical_trade(self, idx: int) -> TradeState:
        """Load a trade from historical data."""
        if self.historical_trades is None:
            return self._generate_synthetic_trade()

        trade = self.historical_trades.iloc[idx]

        return TradeState(
            direction=1.0 if trade.get('quantity', 0) > 0 else -1.0,
            size_pct=abs(trade.get('position_pct', 0.25)),
            days_held=0,
            predicted_return=trade.get('predicted_return', 0.0),
            entry_price=trade.get('fill_price', 100.0),
            current_price=trade.get('fill_price', 100.0),
            unrealized_pnl_pct=0.0,
            max_drawdown_pct=0.0,
            peak_pnl_pct=0.0,
            pnl_velocity=0.0,
            sector_roc=trade.get('sector_roc', 0.0),
            volatility_ratio=trade.get('volatility_ratio', 1.0),
            price_vs_sma=trade.get('price_vs_sma', 0.0),
            margin_usage=self.initial_margin_usage,
            portfolio_heat=trade.get('portfolio_heat', 0.4),
            is_penny_stock=trade.get('fill_price', 100.0) < 1.0,
            stock_price=trade.get('fill_price', 100.0),
        )

    def _get_step_info(self, prev_pnl: float, prev_position: float) -> Dict:
        """Build info dictionary for reward calculation."""
        state = self._trade_state

        return {
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
            'trade_closed': state.days_held >= self.hold_duration or self._position_multiplier < 0.01,
            'realized_pnl_pct': state.unrealized_pnl_pct if state.days_held >= self.hold_duration else 0.0,
            'days_held': state.days_held,
        }

    def _is_done(self, info: Dict) -> bool:
        """Check if episode should terminate."""
        # Margin call = catastrophic end
        if info['margin_call']:
            return True

        # Trade closed (held for full duration or exited)
        if info['trade_closed']:
            return True

        # Max steps reached
        if self._current_step >= self.max_steps:
            return True

        return False

    def render(self, mode: str = 'human'):
        """Render current state (for debugging)."""
        if self._trade_state is None:
            print("No active trade")
            return

        state = self._trade_state
        print(f"\n{'='*50}")
        print(f"Step {self._current_step}/{self.max_steps}")
        print(f"{'='*50}")
        print(f"Direction: {'LONG' if state.direction > 0 else 'SHORT'}")
        print(f"Position Multiplier: {self._position_multiplier:.2%}")
        print(f"Days Held: {state.days_held}/{self.hold_duration}")
        print(f"Price: ${state.current_price:.4f} (Entry: ${state.entry_price:.4f})")
        print(f"Unrealized P&L: {state.unrealized_pnl_pct:.2%}")
        print(f"Max Drawdown: {state.max_drawdown_pct:.2%}")
        print(f"Margin Usage: {state.margin_usage:.2%}")
        print(f"Penny Stock: {state.is_penny_stock}")
        print(f"{'='*50}")


class VectorizedTradingEnv:
    """
    Vectorized environment for parallel training.

    Runs multiple TradingRiskEnv instances in parallel for faster training.
    """

    def __init__(
        self,
        num_envs: int = 8,
        **env_kwargs
    ):
        """
        Initialize vectorized environment.

        Args:
            num_envs: Number of parallel environments
            **env_kwargs: Arguments passed to TradingRiskEnv
        """
        self.num_envs = num_envs
        self.envs = [TradingRiskEnv(**env_kwargs) for _ in range(num_envs)]

        self.state_dim = self.envs[0].state_dim
        self.action_dim = self.envs[0].action_dim

    def reset(self) -> np.ndarray:
        """Reset all environments."""
        states = [env.reset() for env in self.envs]
        return np.stack(states)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments.

        Args:
            actions: Array of shape (num_envs, action_dim)

        Returns:
            Tuple of (states, rewards, dones, infos)
        """
        results = [env.step(a[0]) for env, a in zip(self.envs, actions)]

        states = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]

        # Auto-reset done environments
        for i, done in enumerate(dones):
            if done:
                states[i] = self.envs[i].reset()

        return states, rewards, dones, infos


if __name__ == "__main__":
    # Test the environment
    print("Testing TradingRiskEnv...")

    env = TradingRiskEnv(synthetic_mode=True)

    # Run a few episodes
    for episode in range(3):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0

        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}")
        print(f"{'='*60}")

        while not done:
            # Random action
            action = np.random.uniform(-1, 1)

            next_state, reward, done, info = env.step(action)

            total_reward += reward
            step += 1

            if step <= 3 or done:
                print(f"  Step {step}: action={action:.2f}, "
                      f"reward={reward:.4f}, done={done}")
                if info.get('margin_call'):
                    print("  ⚠️  MARGIN CALL!")

            state = next_state

        print(f"\n  Episode finished: {step} steps, total reward: {total_reward:.4f}")

    # Test vectorized environment
    print("\n" + "="*60)
    print("Testing VectorizedTradingEnv...")
    print("="*60)

    vec_env = VectorizedTradingEnv(num_envs=4)
    states = vec_env.reset()
    print(f"  Vectorized states shape: {states.shape}")

    actions = np.random.uniform(-1, 1, (4, 1))
    next_states, rewards, dones, infos = vec_env.step(actions)
    print(f"  Vectorized step: rewards={rewards}")

    print("\n Trading environment working correctly!")
