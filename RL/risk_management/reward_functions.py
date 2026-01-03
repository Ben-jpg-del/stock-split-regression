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
