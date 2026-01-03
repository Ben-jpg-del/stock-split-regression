import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from collections import deque


@dataclass
class RiskRewardConfig:
    """Configuration for risk management rewards."""

    # ========== CATASTROPHIC LOSS PREVENTION ==========
    margin_call_penalty: float = -100.0      # Absolute disaster
    max_drawdown_penalty: float = -50.0      # Large single-trade loss
    drawdown_threshold: float = 0.20         # 20% drawdown triggers penalty

    # ========== RISK-ADJUSTED RETURNS ==========
    profit_scale: float = 10.0               # Reward per unit profit
    loss_scale: float = 20.0                 # Penalty per unit loss (2x profit)

    # ========== POSITION MANAGEMENT ==========
    early_exit_bonus: float = 5.0            # Reward for exiting losers early
    overtrading_penalty: float = -0.01       # Penalty per transaction cost unit

    # ========== PENNY STOCK RISK ==========
    penny_stock_penalty: float = -5.0        # Penalty for large penny positions
    penny_stock_threshold: float = 1.0       # Price below which stock is "penny"
    max_penny_position_pct: float = 0.10     # Max 10% in any penny stock

    # ========== SHAPING REWARDS ==========
    drawdown_reduction_bonus: float = 1.0    # Bonus for reducing in drawdown
    high_vol_full_position_penalty: float = -2.0  # Penalty for full pos in high vol
    volatility_threshold: float = 1.5        # Vol ratio above which is "high"

    # ========== TRANSACTION COSTS ==========
    transaction_cost_rate: float = 0.001     # 0.1% per trade


@dataclass
class SimpleRewardConfig:
    """Simplified reward configuration - fewer hyperparameters, cleaner signal."""

    # Core rewards (only 3 parameters!)
    margin_call_penalty: float = -10.0       # Terminal penalty (normalized scale)
    pnl_scale: float = 1.0                   # Scale for P&L reward
    transaction_cost: float = 0.001          # 0.1% per position change


class SACRiskReward:
    """
    RECOMMENDED REWARD FUNCTION FOR SAC

    Combines:
    1. Asymmetric loss penalty (from aihedging pattern)
    2. Risk-adjusted returns (Sharpe-like)
    3. Catastrophic event prevention
    4. Transaction cost awareness

    The reward function is designed to:
    - Heavily penalize margin calls (-100)
    - Penalize losses 2x more than reward gains
    - Encourage reducing positions during drawdowns
    - Discourage large positions in penny stocks
    - Discourage overtrading via transaction cost penalty
    """

    def __init__(self, config: Optional[RiskRewardConfig] = None):
        """
        Initialize the reward function.

        Args:
            config: Reward configuration. Uses defaults if None.
        """
        self.config = config or RiskRewardConfig()

        # Rolling window for Sharpe-like calculations
        self.returns_buffer = deque(maxlen=20)
        self.risk_free_rate = 0.05 / 252  # Daily risk-free rate

    def reset(self):
        """Reset the reward function state for a new episode."""
        self.returns_buffer.clear()

    def calculate_reward(
        self,
        state: np.ndarray,
        action: float,
        next_state: np.ndarray,
        info: Dict
    ) -> float:
        """
        Calculate reward for a single step.

        The reward is computed as:
        1. Base P&L reward (asymmetric)
        2. Catastrophic loss penalties
        3. Position management bonuses/penalties
        4. Transaction cost penalty

        Args:
            state: Current state vector (12 dims)
            action: Continuous action in [-1, 1]
            next_state: Next state vector (12 dims)
            info: Dictionary containing:
                - margin_call: bool
                - pnl_change: float (unrealized P&L change)
                - wealth: float (current portfolio value normalized)
                - max_drawdown_pct: float
                - new_position: float (new position size)
                - old_position: float (old position size)
                - is_penny_stock: bool
                - position_pct: float (position as % of portfolio)
                - volatility_ratio: float (current vol / historical vol)
                - trade_closed: bool (optional)
                - realized_pnl_pct: float (optional, if trade closed)
                - days_held: int (optional)

        Returns:
            float: Scalar reward value
        """
        reward = 0.0
        cfg = self.config

        # ===== 1. CATASTROPHIC LOSS PREVENTION =====
        if info.get('margin_call', False):
            return cfg.margin_call_penalty  # Immediate termination

        # ===== 2. P&L-BASED REWARD (ASYMMETRIC) =====
        pnl_change = info.get('pnl_change', 0.0)
        wealth = max(info.get('wealth', 1.0), 0.1)  # Avoid div by zero

        # Track returns for potential Sharpe calculation
        self.returns_buffer.append(pnl_change)

        if pnl_change >= 0:
            # Reward gains (smaller weight)
            reward += (pnl_change / wealth) * cfg.profit_scale
        else:
            # Penalize losses (larger weight) - asymmetric penalty
            penalty = -pnl_change / wealth
            reward -= penalty * cfg.loss_scale

        # ===== 3. DRAWDOWN SHAPING =====
        max_dd = info.get('max_drawdown_pct', 0.0)
        if max_dd > cfg.drawdown_threshold:
            excess_dd = max_dd - cfg.drawdown_threshold
            reward += excess_dd * cfg.max_drawdown_penalty

        # ===== 4. TRANSACTION COST =====
        old_position = info.get('old_position', 0.0)
        new_position = info.get('new_position', 0.0)
        position_change = abs(new_position - old_position)
        reward -= position_change * cfg.transaction_cost_rate * abs(cfg.overtrading_penalty) * 100

        # ===== 5. PENNY STOCK RISK =====
        is_penny = info.get('is_penny_stock', False)
        position_pct = info.get('position_pct', 0.0)

        if is_penny and position_pct > cfg.max_penny_position_pct:
            reward += cfg.penny_stock_penalty

        # ===== 6. ACTION-AWARE SHAPING =====
        position_mult = (action + 1) / 2  # Map [-1, 1] to [0, 1]

        # Reward reducing position when in drawdown
        if max_dd > 0.10 and position_mult < 0.5:
            reward += cfg.drawdown_reduction_bonus

        # Penalize holding full position during high volatility
        vol_ratio = info.get('volatility_ratio', 1.0)
        if vol_ratio > cfg.volatility_threshold and position_mult > 0.8:
            reward += cfg.high_vol_full_position_penalty

        # ===== 7. TRADE CLOSE BONUS =====
        if info.get('trade_closed', False):
            realized_pnl = info.get('realized_pnl_pct', 0.0)
            days_held = info.get('days_held', 3)

            # Bonus for closing losers early
            if realized_pnl < 0 and days_held < 2:
                reward += cfg.early_exit_bonus

        return reward

    def get_sharpe_bonus(self) -> float:
        """
        Calculate Sharpe-like bonus from rolling returns.

        Returns:
            float: Sharpe ratio bonus (0 if insufficient data)
        """
        if len(self.returns_buffer) < 10:
            return 0.0

        returns = np.array(self.returns_buffer)
        excess_returns = returns - self.risk_free_rate

        denom = excess_returns.std()
        if denom > 1e-8:
            sharpe = excess_returns.mean() / denom
            return sharpe * 5.0  # Scale factor
        return 0.0


class SparseRiskReward(SACRiskReward):
    """
    Sparse reward variant - only rewards on trade close or margin call.

    Use this for environments where you want less frequent but more
    meaningful reward signals.
    """

    def calculate_reward(
        self,
        state: np.ndarray,
        action: float,
        next_state: np.ndarray,
        info: Dict
    ) -> float:
        """Calculate sparse reward (only on episode-ending events)."""
        cfg = self.config

        # Catastrophic loss
        if info.get('margin_call', False):
            return cfg.margin_call_penalty

        # Trade closed - give final P&L reward
        if info.get('trade_closed', False):
            realized_pnl = info.get('realized_pnl_pct', 0.0)

            if realized_pnl >= 0:
                reward = realized_pnl * cfg.profit_scale * 10
            else:
                reward = realized_pnl * cfg.loss_scale * 10

            # Early exit bonus
            days_held = info.get('days_held', 3)
            if realized_pnl < 0 and days_held < 2:
                reward += cfg.early_exit_bonus

            return reward

        # No reward for intermediate steps
        return 0.0


class DenseRiskReward(SACRiskReward):
    """
    Dense reward variant with additional shaping signals.

    Provides more frequent feedback for faster learning,
    but may introduce some reward hacking opportunities.
    """

    def __init__(self, config: Optional[RiskRewardConfig] = None):
        super().__init__(config)
        self.step_penalty = -0.01  # Small penalty per timestep

    def calculate_reward(
        self,
        state: np.ndarray,
        action: float,
        next_state: np.ndarray,
        info: Dict
    ) -> float:
        """Calculate dense reward with step penalty."""
        # Get base reward
        reward = super().calculate_reward(state, action, next_state, info)

        # Add step penalty (encourages decisive action)
        reward += self.step_penalty

        # Add Sharpe bonus if enough data
        reward += self.get_sharpe_bonus()

        # Velocity-based shaping (reward improving momentum)
        pnl_velocity = info.get('pnl_velocity', 0.0)
        if pnl_velocity > 0:
            reward += 0.5  # Positive momentum bonus
        elif pnl_velocity < -0.05:
            reward -= 1.0  # Accelerating losses penalty

        return reward


class CurriculumRiskReward(SACRiskReward):
    def __init__(
        self,
        config: Optional[RiskRewardConfig] = None,
        curriculum_episodes: int = 10000
    ):
        super().__init__(config)
        if curriculum_episodes <= 0:
            raise ValueError(
                f"curriculum_episodes must be a positive integer, got {curriculum_episodes}"
            )
        self.curriculum_episodes = curriculum_episodes
        self.current_episode = 0

    def set_episode(self, episode: int):
        """Set current episode for curriculum progression."""
        self.current_episode = episode

    @property
    def curriculum_stage(self) -> int:
        """Get current curriculum stage (1, 2, or 3)."""
        if self.curriculum_episodes <= 0:
            return 1  # Defensive fallback
        progress = self.current_episode / self.curriculum_episodes
        if progress < 0.33:
            return 1
        elif progress < 0.66:
            return 2
        return 3

    def calculate_reward(
        self,
        state: np.ndarray,
        action: float,
        next_state: np.ndarray,
        info: Dict
    ) -> float:
        """Calculate curriculum-aware reward."""
        cfg = self.config
        stage = self.curriculum_stage
        reward = 0.0

        # Stage 1: Only catastrophic events
        if info.get('margin_call', False):
            return cfg.margin_call_penalty

        if stage == 1:
            # Only margin call penalty in stage 1
            return reward

        # Stage 2+: Add P&L rewards
        pnl_change = info.get('pnl_change', 0.0)
        wealth = max(info.get('wealth', 1.0), 0.1)

        if pnl_change >= 0:
            reward += (pnl_change / wealth) * cfg.profit_scale
        else:
            reward -= (-pnl_change / wealth) * cfg.loss_scale

        if stage == 2:
            return reward

        # Stage 3: Full reward with all shaping
        return super().calculate_reward(state, action, next_state, info)


class SimpleRiskReward:
    """
    SIMPLIFIED REWARD FUNCTION

    Design principles:
    1. Fewer components = cleaner learning signal
    2. Consistent scale across all rewards
    3. Focus on the actual goal: maximize risk-adjusted P&L

    Components (only 3):
    1. P&L change (the actual goal)
    2. Margin call penalty (hard constraint)
    3. Transaction cost (realistic friction)

    No shaping rewards that might create local optima.
    """

    def __init__(self, config: Optional[SimpleRewardConfig] = None):
        self.config = config or SimpleRewardConfig()
        self._episode_pnl = 0.0

    def reset(self):
        """Reset for new episode."""
        self._episode_pnl = 0.0

    def calculate_reward(
        self,
        state: np.ndarray,
        action: float,
        next_state: np.ndarray,
        info: Dict
    ) -> float:
        """
        Simple reward: P&L change - transaction cost - margin penalty.

        All rewards on similar scale (-10 to +10 typical range).
        """
        # 1. Margin call = immediate large penalty and episode end
        if info.get('margin_call', False):
            return self.config.margin_call_penalty

        reward = 0.0

        # 2. P&L-based reward (the actual goal)
        # Use raw P&L change, scaled to reasonable range
        pnl_change = info.get('pnl_change', 0.0)
        reward += pnl_change * self.config.pnl_scale * 100  # Scale to ~[-5, +5] range

        # Track cumulative for logging
        self._episode_pnl += pnl_change

        # 3. Transaction cost (realistic friction, discourages churning)
        old_pos = info.get('old_position', 0.0)
        new_pos = info.get('new_position', 0.0)
        position_change = abs(new_pos - old_pos)
        reward -= position_change * self.config.transaction_cost * 10

        return reward


class PnLOnlyReward:
    """
    Minimal reward: just P&L at episode end.

    Use this as a baseline to see if complex shaping helps or hurts.
    """

    def __init__(self, margin_penalty: float = -10.0):
        self.margin_penalty = margin_penalty
        self._cumulative_pnl = 0.0

    def reset(self):
        self._cumulative_pnl = 0.0

    def calculate_reward(
        self,
        state: np.ndarray,
        action: float,
        next_state: np.ndarray,
        info: Dict
    ) -> float:
        """Reward = final P&L at episode end, 0 otherwise."""
        if info.get('margin_call', False):
            return self.margin_penalty

        pnl_change = info.get('pnl_change', 0.0)
        self._cumulative_pnl += pnl_change

        # Only give reward at episode end
        if info.get('trade_closed', False):
            return self._cumulative_pnl * 100  # Scale to reasonable range

        return 0.0


def create_reward_function(
    reward_type: str = 'simple',
    config=None
):
    """
    Factory function for reward functions.

    Args:
        reward_type: One of 'simple', 'pnl_only', 'sac', 'sparse', 'dense', 'curriculum'
        config: Configuration object (type depends on reward_type)

    Returns:
        Reward function instance
    """
    reward_classes = {
        'simple': (SimpleRiskReward, SimpleRewardConfig),
        'pnl_only': (PnLOnlyReward, None),
        'sac': (SACRiskReward, RiskRewardConfig),
        'sparse': (SparseRiskReward, RiskRewardConfig),
        'dense': (DenseRiskReward, RiskRewardConfig),
        'curriculum': (CurriculumRiskReward, RiskRewardConfig),
    }

    if reward_type not in reward_classes:
        raise ValueError(f"Unknown reward type: {reward_type}. "
                        f"Choose from {list(reward_classes.keys())}")

    cls, default_config_cls = reward_classes[reward_type]

    if reward_type == 'pnl_only':
        return cls() if config is None else cls(config)

    return cls(config)


if __name__ == "__main__":
    # Test all reward functions
    state = np.random.randn(12).astype(np.float32)
    next_state = np.random.randn(12).astype(np.float32)

    # Standard test info
    info_profit = {
        'margin_call': False,
        'pnl_change': 0.02,
        'wealth': 1.05,
        'max_drawdown_pct': 0.08,
        'new_position': 0.8,
        'old_position': 1.0,
        'is_penny_stock': False,
        'position_pct': 0.25,
        'volatility_ratio': 1.0,
        'trade_closed': False,
    }

    info_loss = {
        'margin_call': False,
        'pnl_change': -0.05,
        'wealth': 0.95,
        'max_drawdown_pct': 0.15,
        'new_position': 0.5,
        'old_position': 1.0,
        'is_penny_stock': True,
        'position_pct': 0.15,
        'volatility_ratio': 2.0,
        'trade_closed': False,
    }

    info_margin = {'margin_call': True}

    info_close = {
        'margin_call': False,
        'pnl_change': 0.01,
        'wealth': 1.10,
        'new_position': 0.0,
        'old_position': 0.5,
        'trade_closed': True,
    }

    print("=" * 60)
    print("REWARD FUNCTION COMPARISON")
    print("=" * 60)

    # Test each reward type
    for rtype in ['simple', 'pnl_only', 'sac', 'sparse']:
        rf = create_reward_function(rtype)
        rf.reset()

        r_profit = rf.calculate_reward(state, 0.6, next_state, info_profit)
        rf.reset()
        r_loss = rf.calculate_reward(state, -0.5, next_state, info_loss)
        rf.reset()
        r_margin = rf.calculate_reward(state, 0.0, next_state, info_margin)
        rf.reset()
        r_close = rf.calculate_reward(state, -1.0, next_state, info_close)

        print(f"\n{rtype.upper()} ({rf.__class__.__name__}):")
        print(f"  Profit (+2% P&L):  {r_profit:+8.4f}")
        print(f"  Loss (-5% P&L):    {r_loss:+8.4f}")
        print(f"  Margin call:       {r_margin:+8.4f}")
        print(f"  Trade close:       {r_close:+8.4f}")

    print("\n" + "=" * 60)
    print("RECOMMENDATION: Use 'simple' for cleaner learning signal")
    print("=" * 60)
