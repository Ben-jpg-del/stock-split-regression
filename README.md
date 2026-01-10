# Stock Split Trading Algorithm

A machine learning-based trading strategy that predicts returns around stock split events using sector momentum and split factors. The strategy employs a **Soft Actor-Critic (SAC) reinforcement learning agent** for dynamic position sizing and risk management.

## RL Policy: Logic, Mechanics & Reward Functions

### Overview

The RL policy acts as a risk management overlay that dynamically adjusts position sizes based on market conditions and portfolio state. Rather than making entry/exit decisions, it learns **when to reduce exposure** to protect capital while letting winners run.

### State Space (11 Features)

The agent observes an 11-dimensional state vector at each step:

| Feature | Description | Range |
|---------|-------------|-------|
| `direction` | Trade direction | 1.0 (long) / -1.0 (short) |
| `size_pct` | Position size as % of portfolio | 0.1 - 0.3 |
| `days_held` | Days in trade (normalized by 3) | 0.0+ |
| `unrealized_pnl_pct` | Current P&L percentage | unbounded |
| `max_drawdown_pct` | Maximum drawdown from peak | 0.0+ |
| `pnl_velocity` | Rate of P&L change (dP&L/dt) | unbounded |
| `sector_roc` | Sector momentum (XLK Rate of Change) | unbounded |
| `volatility_ratio` | Current vol / historical vol | 0.5 - 3.0 |
| `price_vs_sma` | Price deviation from SMA | unbounded |
| `margin_usage` | Margin utilization | 0.1 - 1.0 |
| `portfolio_heat` | Overall portfolio risk concentration | 0.0 - 1.0 |

### Action Space

The SAC agent outputs a **continuous action in [-1, 1]** which is transformed into a position multiplier:

```
action ∈ [-1, 1] (tanh-squashed Gaussian policy output)
    ↓
position_multiplier = (action + 1) / 2  ∈ [0, 1]
    ↓
In production: scaled_multiplier = 0.3 + raw_multiplier * 0.7  ∈ [0.3, 1.0]
    ↓
adjusted_quantity = base_quantity × scaled_multiplier
```

The 0.3 floor prevents the agent from becoming overly conservative.

### Reward Function: RiskManagementReward (Primary)

The recommended reward function uses **trajectory-based learning** to teach proper risk management without directly rewarding P&L:

#### P&L Trajectory Detection
- **Improving**: P&L getting better over last 3 steps → reward letting winners run
- **Deteriorating**: P&L getting worse → reward cutting positions
- **Flat**: Neutral trajectory → neutral rewards

#### SACRiskReward

A more traditional per-step reward function:
- **Profit scaling**: `+10.0 × (pnl_change / wealth)`
- **Loss scaling**: `-20.0 × (pnl_change / wealth)` (2× asymmetry)
- **Drawdown penalties**: Scaled based on excess above 20% threshold
- **Transaction costs**: `-0.01 × position_change × 100`
- **Penny stock penalty**: `-5.0` if >10% in stocks <$1

### SAC Algorithm Details

**Actor Network (Gaussian Policy)**:
- Architecture: 11 → 256 → 256 → 1 (mean) + 1 (log_std)
- Activation: ReLU hidden, tanh output squashing
- Log-std clamped to [-2, 2] for stable exploration

**Critic Network (Dual Q-functions)**:
- Two Q-networks for stability (take minimum)
- Architecture: (11 + 1) → 256 → 256 → 1
- Target networks with soft update: τ = 0.005

**Hyperparameters**:
- Learning rates: actor=3e-4, critic=3e-4, alpha=3e-5
- Discount factor (γ): 0.99
- Batch size: 256
- Replay buffer: 1M transitions
- Target entropy: -1.0 (for 1-D action space)
- Alpha (temperature): learned, clamped min=0.01

## Strategy Results
<img width="1466" height="810" alt="Screenshot 2025-12-31 202115" src="https://github.com/user-attachments/assets/cc37fa99-71be-4cf2-b1e0-5b8e571ab553" /> 
<img width="1050" height="355" alt="Screenshot 2025-12-31 212215" src="https://github.com/user-attachments/assets/eedd36e5-4ce4-463d-b56a-6fb80450877f" />
<img width="944" height="328" alt="Screenshot 2025-12-31 212233" src="https://github.com/user-attachments/assets/c81b4db9-c312-4e76-a65a-c810bcd61f96" />


## Repository Structure

```
mcpt-main/
├── quantconnect/           # QuantConnect algorithm
│   └── main.py             # Main algorithm for QC platform
├── standalone/             # Standalone paper trading
│   └── paper_trading.py    # Alpaca + Polygon.io implementation
├── research/               # Research and analysis
│   ├── research_quantbook.py
│   ├── bar_permute.py
│   └── mcpt_skeleton.py
├── logs/                   # Log files (gitignored)
├── .env.example            # Environment template
├── requirements.txt        # Python dependencies
└── README.md
```
