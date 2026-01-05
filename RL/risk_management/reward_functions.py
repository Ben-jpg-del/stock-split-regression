import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from collections import deque


@dataclass
class RiskRewardConfig:
    """Configuration for risk management rewards."""

    margin_call_penalty: float = -100.0     
    max_drawdown_penalty: float = -50.0     
    drawdown_threshold: float = 0.20         

    profit_scale: float = 10.0              
    loss_scale: float = 20.0                 

    early_exit_bonus: float = 5.0            
    overtrading_penalty: float = -0.01       

    penny_stock_penalty: float = -5.0       
    penny_stock_threshold: float = 1.0       
    max_penny_position_pct: float = 0.10     
    drawdown_reduction_bonus: float = 1.0    
    high_vol_full_position_penalty: float = -2.0  
    volatility_threshold: float = 1.5        
    transaction_cost_rate: float = 0.001     


@dataclass
class SimpleRewardConfig:

    margin_call_penalty: float = -10.0
    pnl_scale: float = 1.0
    transaction_cost: float = 0.001


@dataclass
class RiskManagementConfig:
    """Configuration for trajectory-based risk management rewards.

    Key principle: Reward APPROPRIATE risk decisions, not raw P&L.
    The agent should learn WHEN to reduce/increase positions based on
    risk signals, not that "all positions are bad."
    """
    # Catastrophic event penalty
    margin_call_penalty: float = -50.0

    # Trajectory-based rewards (the core of risk management)
    cut_loser_bonus: float = 2.0           # Reward for reducing deteriorating positions
    hold_loser_penalty: float = -0.5       # Per-step penalty for holding losing positions
    let_winner_run_bonus: float = 1.0      # Reward for maintaining winning positions
    conviction_bonus: float = 0.5          # Bonus for full position on improving trajectory

    # Risk signal response rewards
    drawdown_response_bonus: float = 1.5   # Reward for reducing during drawdown
    volatility_response_bonus: float = 1.0 # Reward for reducing during high volatility

    # Anti-whipsaw (prevent noisy position changes)
    whipsaw_penalty: float = -0.3          # Penalty for rapid position changes
    whipsaw_threshold: float = 0.3         # Position change that triggers penalty

    # Trajectory detection thresholds
    trajectory_window: int = 3             # Steps to consider for trajectory
    deteriorating_threshold: float = -0.01 # P&L change to count as deteriorating         


class SACRiskReward:
    def __init__(self, config: Optional[RiskRewardConfig] = None):
        self.config = config or RiskRewardConfig()

        self.returns_buffer = deque(maxlen=20)
        self.risk_free_rate = 0.05 / 252  

    def reset(self):
        self.returns_buffer.clear()

    def calculate_reward(
        self,
        state: np.ndarray,
        action: float,
        next_state: np.ndarray,
        info: Dict
    ) -> float:
        reward = 0.0
        cfg = self.config

        if info.get('margin_call', False):
            return cfg.margin_call_penalty  

        pnl_change = info.get('pnl_change', 0.0)
        wealth = max(info.get('wealth', 1.0), 0.1)  

        self.returns_buffer.append(pnl_change)

        if pnl_change >= 0:
            reward += (pnl_change / wealth) * cfg.profit_scale
        else:
            penalty = -pnl_change / wealth
            reward -= penalty * cfg.loss_scale

        max_dd = info.get('max_drawdown_pct', 0.0)
        if max_dd > cfg.drawdown_threshold:
            excess_dd = max_dd - cfg.drawdown_threshold
            reward += excess_dd * cfg.max_drawdown_penalty

        old_position = info.get('old_position', 0.0)
        new_position = info.get('new_position', 0.0)
        position_change = abs(new_position - old_position)
        reward -= position_change * cfg.transaction_cost_rate * abs(cfg.overtrading_penalty) * 100

        is_penny = info.get('is_penny_stock', False)
        position_pct = info.get('position_pct', 0.0)

        if is_penny and position_pct > cfg.max_penny_position_pct:
            reward += cfg.penny_stock_penalty

        position_mult = (action + 1) / 2  

        if max_dd > 0.10 and position_mult < 0.5:
            reward += cfg.drawdown_reduction_bonus

        vol_ratio = info.get('volatility_ratio', 1.0)
        if vol_ratio > cfg.volatility_threshold and position_mult > 0.8:
            reward += cfg.high_vol_full_position_penalty

        if info.get('trade_closed', False):
            realized_pnl = info.get('realized_pnl_pct', 0.0)
            days_held = info.get('days_held', 3)

            if realized_pnl < 0 and days_held < 2:
                reward += cfg.early_exit_bonus

        return reward

    def get_sharpe_bonus(self) -> float:
        if len(self.returns_buffer) < 10:
            return 0.0

        returns = np.array(self.returns_buffer)
        excess_returns = returns - self.risk_free_rate

        denom = excess_returns.std()
        if denom > 1e-8:
            sharpe = excess_returns.mean() / denom
            return sharpe * 5.0  
        return 0.0


class SparseRiskReward(SACRiskReward):
    def calculate_reward(
        self,
        state: np.ndarray,
        action: float,
        next_state: np.ndarray,
        info: Dict
    ) -> float:
        cfg = self.config

        if info.get('margin_call', False):
            return cfg.margin_call_penalty

        if info.get('trade_closed', False):
            realized_pnl = info.get('realized_pnl_pct', 0.0)

            if realized_pnl >= 0:
                reward = realized_pnl * cfg.profit_scale * 10
            else:
                reward = realized_pnl * cfg.loss_scale * 10

            days_held = info.get('days_held', 3)
            if realized_pnl < 0 and days_held < 2:
                reward += cfg.early_exit_bonus

            return reward

        return 0.0


class DenseRiskReward(SACRiskReward):
    def __init__(self, config: Optional[RiskRewardConfig] = None):
        super().__init__(config)
        self.step_penalty = -0.01  

    def calculate_reward(
        self,
        state: np.ndarray,
        action: float,
        next_state: np.ndarray,
        info: Dict
    ) -> float:
        reward = super().calculate_reward(state, action, next_state, info)

        reward += self.step_penalty

        reward += self.get_sharpe_bonus()

        pnl_velocity = info.get('pnl_velocity', 0.0)
        if pnl_velocity > 0:
            reward += 0.5  
        elif pnl_velocity < -0.05:
            reward -= 1.0  

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
        self.current_episode = episode

    @property
    def curriculum_stage(self) -> int:
        if self.curriculum_episodes <= 0:
            return 1  
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
        cfg = self.config
        stage = self.curriculum_stage
        reward = 0.0

        if info.get('margin_call', False):
            return cfg.margin_call_penalty

        if stage == 1:
            return reward

        pnl_change = info.get('pnl_change', 0.0)
        wealth = max(info.get('wealth', 1.0), 0.1)

        if pnl_change >= 0:
            reward += (pnl_change / wealth) * cfg.profit_scale
        else:
            reward -= (-pnl_change / wealth) * cfg.loss_scale

        if stage == 2:
            return reward

        return super().calculate_reward(state, action, next_state, info)


class SimpleRiskReward:
    def __init__(self, config: Optional[SimpleRewardConfig] = None):
        self.config = config or SimpleRewardConfig()
        self._episode_pnl = 0.0

    def reset(self):
        self._episode_pnl = 0.0

    def calculate_reward(
        self,
        state: np.ndarray,
        action: float,
        next_state: np.ndarray,
        info: Dict
    ) -> float:
        if info.get('margin_call', False):
            return self.config.margin_call_penalty

        reward = 0.0

        pnl_change = info.get('pnl_change', 0.0)
        reward += pnl_change * self.config.pnl_scale * 100  

        self._episode_pnl += pnl_change

        old_pos = info.get('old_position', 0.0)
        new_pos = info.get('new_position', 0.0)
        position_change = abs(new_pos - old_pos)
        reward -= position_change * self.config.transaction_cost * 10

        return reward


class PnLOnlyReward:
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
        if info.get('margin_call', False):
            return self.margin_penalty

        pnl_change = info.get('pnl_change', 0.0)
        self._cumulative_pnl += pnl_change

        if info.get('trade_closed', False):
            return self._cumulative_pnl * 100

        return 0.0


class RiskManagementReward:
    """Trajectory-based risk management reward function.

    Key design principles:
    1. NO direct P&L rewards - the agent shouldn't chase profits
    2. Reward based on TRAJECTORY - cut losers, let winners run
    3. Allow full position (multiplier=1.0) when conditions are favorable
    4. Only penalize BAD risk decisions, not position-taking itself

    This prevents the agent from learning "avoid all positions" and instead
    teaches it WHEN to reduce positions based on risk signals.
    """

    def __init__(self, config: Optional[RiskManagementConfig] = None):
        self.config = config or RiskManagementConfig()
        self.pnl_history = deque(maxlen=self.config.trajectory_window)
        self.position_history = deque(maxlen=3)

    def reset(self):
        self.pnl_history.clear()
        self.position_history.clear()

    def _get_trajectory(self) -> str:
        """Determine P&L trajectory: 'improving', 'deteriorating', or 'flat'."""
        if len(self.pnl_history) < 2:
            return 'flat'

        recent = list(self.pnl_history)
        # Compare end to beginning
        pnl_change = recent[-1] - recent[0]

        if pnl_change > self.config.deteriorating_threshold:
            return 'improving'
        elif pnl_change < self.config.deteriorating_threshold:
            return 'deteriorating'
        return 'flat'

    def calculate_reward(
        self,
        state: np.ndarray,
        action: float,
        next_state: np.ndarray,
        info: Dict
    ) -> float:
        """Calculate reward based on risk management quality, not raw P&L."""
        reward = 0.0
        cfg = self.config

        # Margin call is catastrophic - strong penalty
        if info.get('margin_call', False):
            return cfg.margin_call_penalty

        # Get current state
        position_mult = (action + 1) / 2  # Map [-1,1] to [0,1]
        unrealized_pnl = info.get('unrealized_pnl_pct', 0.0)
        old_position = info.get('old_position', 1.0)
        max_drawdown = info.get('max_drawdown_pct', 0.0)
        volatility_ratio = info.get('volatility_ratio', 1.0)

        # Track P&L history for trajectory
        self.pnl_history.append(unrealized_pnl)
        trajectory = self._get_trajectory()

        # ============================================================
        # CORE RISK MANAGEMENT REWARDS
        # ============================================================

        # 1. DETERIORATING POSITION (P&L getting worse)
        if trajectory == 'deteriorating':
            if position_mult < old_position:
                # GOOD: Cutting a losing position
                reduction = old_position - position_mult
                reward += cfg.cut_loser_bonus * reduction
            else:
                # BAD: Holding or increasing a deteriorating position
                reward += cfg.hold_loser_penalty

        # 2. IMPROVING POSITION (P&L getting better)
        elif trajectory == 'improving':
            if position_mult >= 0.7:
                # GOOD: Letting winner run with conviction
                reward += cfg.let_winner_run_bonus
                if position_mult >= 0.9:
                    # Extra bonus for full conviction on winners
                    reward += cfg.conviction_bonus

        # 3. UNDERWATER POSITION (current P&L is negative)
        if unrealized_pnl < 0:
            # Only penalize if not actively cutting
            if position_mult >= old_position:
                reward += cfg.hold_loser_penalty * 0.5

        # ============================================================
        # RISK SIGNAL RESPONSE REWARDS
        # ============================================================

        # 4. DRAWDOWN RESPONSE
        if max_drawdown > 0.10:  # 10% drawdown threshold
            if position_mult < 0.5:
                # GOOD: Reducing exposure during significant drawdown
                reward += cfg.drawdown_response_bonus
            elif position_mult > 0.8:
                # BAD: Full position during high drawdown
                reward -= cfg.drawdown_response_bonus * 0.5

        # 5. VOLATILITY RESPONSE
        if volatility_ratio > 1.5:  # High volatility
            if position_mult < 0.7:
                # GOOD: Reducing during high volatility
                reward += cfg.volatility_response_bonus
            elif position_mult > 0.9:
                # BAD: Full position during high volatility
                reward -= cfg.volatility_response_bonus * 0.3

        # ============================================================
        # ANTI-WHIPSAW (prevent noisy trading)
        # ============================================================

        position_change = abs(position_mult - old_position)
        if position_change > cfg.whipsaw_threshold:
            reward += cfg.whipsaw_penalty

        # ============================================================
        # TRADE OUTCOME BONUS (sparse reward at trade end)
        # ============================================================

        if info.get('trade_closed', False):
            realized_pnl = info.get('realized_pnl_pct', 0.0)
            days_held = info.get('days_held', 3)

            # Bonus for exiting a loser early
            if realized_pnl < 0 and days_held < 2:
                reward += 3.0

            # Bonus for riding a winner
            if realized_pnl > 0.05 and days_held >= 2:
                reward += 2.0

        return reward


def create_reward_function(
    reward_type: str = 'simple',
    config=None
):
    reward_classes = {
        'simple': (SimpleRiskReward, SimpleRewardConfig),
        'pnl_only': (PnLOnlyReward, None),
        'sac': (SACRiskReward, RiskRewardConfig),
        'sparse': (SparseRiskReward, RiskRewardConfig),
        'dense': (DenseRiskReward, RiskRewardConfig),
        'curriculum': (CurriculumRiskReward, RiskRewardConfig),
        'risk_management': (RiskManagementReward, RiskManagementConfig),
    }

    if reward_type not in reward_classes:
        raise ValueError(f"Unknown reward type: {reward_type}. "
                        f"Choose from {list(reward_classes.keys())}")

    cls, default_config_cls = reward_classes[reward_type]

    if reward_type == 'pnl_only':
        return cls() if config is None else cls(config)

    return cls(config)


if __name__ == "__main__":
    state = np.random.randn(12).astype(np.float32)
    next_state = np.random.randn(12).astype(np.float32)

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

    for rtype in ['simple', 'pnl_only', 'sac', 'sparse', 'risk_management']:
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
    print("RECOMMENDATION: Use 'risk_management' for proper position sizing")
    print("  - Rewards appropriate risk decisions, not raw P&L")
    print("  - Allows full position when trajectory is improving")
    print("  - Penalizes holding losers, rewards cutting them early")
    print("=" * 60)
