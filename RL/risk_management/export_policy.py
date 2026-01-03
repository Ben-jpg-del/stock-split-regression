"""
Export trained SAC policy for QuantConnect deployment.

The exported policy is a simple numpy-based inference that doesn't require PyTorch.
Upload the generated JSON file to QuantConnect Object Store.

Usage:
    python -m RL.risk_management.export_policy --checkpoint checkpoints/sac_best.pt
"""

import argparse
import json
import numpy as np
from pathlib import Path


def export_policy_to_json(checkpoint_path: str, output_path: str = None):
    """
    Export SAC actor network weights to JSON for QuantConnect.

    The exported format includes all weights needed to run inference
    using only numpy (no PyTorch required).
    """
    import torch

    print(f"Loading checkpoint: {checkpoint_path}")
    # weights_only=True prevents arbitrary code execution via pickle (CVE-2024-5480)
    # Safe since checkpoints only contain tensors and basic Python types
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    # Extract actor network state dict
    actor_state = checkpoint['actor_state_dict']

    # Convert to numpy and prepare for JSON serialization
    weights = {}
    for key, tensor in actor_state.items():
        weights[key] = tensor.numpy().tolist()

    # Also save network architecture info
    export_data = {
        'version': '1.0',
        'algorithm': 'SAC',
        'description': 'Risk management policy for stock split strategy',
        'weights': weights,
        'architecture': {
            'state_dim': 12,
            'action_dim': 1,
            'hidden_dims': [256, 256],
            'activation': 'relu',
        },
        'action_scaling': {
            'min': 0.0,  # Position multiplier range
            'max': 1.0,
        },
        'training_info': {
            'episode': checkpoint.get('episode', 'unknown'),
        }
    }

    # Determine output path
    if output_path is None:
        output_path = str(Path(checkpoint_path).parent / 'policy_export.json')

    print(f"Exporting to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(export_data, f)

    # Calculate file size
    file_size = Path(output_path).stat().st_size
    print(f"Export complete: {file_size:,} bytes")

    # Also create a compact version for QC
    if output_path.endswith('.json'):
        compact_path = output_path.replace('.json', '_compact.json')
    else:
        compact_path = output_path + '_compact.json'
    with open(compact_path, 'w') as f:
        json.dump(export_data, f, separators=(',', ':'))

    compact_size = Path(compact_path).stat().st_size
    print(f"Compact version: {compact_path} ({compact_size:,} bytes)")

    return output_path


def create_numpy_inference_code():
    """Generate the numpy-only inference code for QuantConnect."""

    code = '''
# ============================================================
# SAC RISK POLICY - NUMPY INFERENCE (for QuantConnect)
# ============================================================
# This code runs the trained SAC policy using only numpy.
# Copy this into your QuantConnect algorithm.

import numpy as np
import json

class SACRiskPolicy:
    """
    Numpy-only implementation of SAC actor for risk management.
    Outputs position multiplier [0, 1] given trade state.
    """

    def __init__(self, policy_json_path: str = None, policy_dict: dict = None):
        """
        Initialize from JSON file path or dict.

        Args:
            policy_json_path: Path to exported policy JSON
            policy_dict: Pre-loaded policy dict (for QC Object Store)
        """
        if policy_dict is not None:
            self._load_from_dict(policy_dict)
        elif policy_json_path is not None:
            with open(policy_json_path, 'r') as f:
                policy_dict = json.load(f)
            self._load_from_dict(policy_dict)
        else:
            # Default to neutral (multiplier = 1.0, no risk reduction)
            self.weights = None

    def _load_from_dict(self, policy_dict: dict):
        """Load weights from policy dict."""
        self.weights = {}
        for key, value in policy_dict['weights'].items():
            self.weights[key] = np.array(value)
        self.architecture = policy_dict.get('architecture', {})

    def _relu(self, x):
        return np.maximum(0, x)

    def _tanh(self, x):
        return np.tanh(x)

    def get_position_multiplier(self, state: np.ndarray) -> float:
        """
        Get position multiplier from state.

        Args:
            state: 12-dim state vector [direction, size_pct, days_held, ...]

        Returns:
            Position multiplier in [0, 1]
            - 1.0 = full position (no risk reduction)
            - 0.0 = close position entirely
        """
        if self.weights is None:
            return 1.0  # Default: no risk adjustment

        x = state.flatten()

        # Forward pass through actor network
        # fc1
        x = x @ self.weights['fc1.weight'].T + self.weights['fc1.bias']
        x = self._relu(x)

        # fc2
        x = x @ self.weights['fc2.weight'].T + self.weights['fc2.bias']
        x = self._relu(x)

        # mean_head (we use mean for deterministic inference)
        mean = x @ self.weights['mean_head.weight'].T + self.weights['mean_head.bias']

        # Tanh squashing and scale to [0, 1]
        action = self._tanh(mean)
        multiplier = (action + 1) / 2  # Scale from [-1, 1] to [0, 1]

        return float(np.clip(multiplier, 0, 1))

    def should_reduce_position(self, state: np.ndarray, threshold: float = 0.7) -> bool:
        """Check if policy recommends reducing position."""
        return self.get_position_multiplier(state) < threshold

    def should_close_position(self, state: np.ndarray, threshold: float = 0.3) -> bool:
        """Check if policy recommends closing position."""
        return self.get_position_multiplier(state) < threshold


def create_state_vector(
    direction: float,
    size_pct: float,
    days_held: int,
    predicted_return: float,
    unrealized_pnl_pct: float,
    max_drawdown_pct: float,
    pnl_velocity: float,
    sector_roc: float,
    volatility_ratio: float,
    margin_usage: float,
    portfolio_heat: float,
    is_penny_stock: bool,
) -> np.ndarray:
    """
    Create state vector for SAC policy.

    This matches the TradeState used during training.
    """
    return np.array([
        direction,           # 1.0 long, -1.0 short
        size_pct,            # Position size as % of portfolio
        float(days_held),    # Days in trade
        predicted_return,    # Model's predicted return
        unrealized_pnl_pct,  # Current unrealized P&L %
        max_drawdown_pct,    # Max drawdown from peak
        pnl_velocity,        # Rate of P&L change
        sector_roc,          # Sector momentum (XLK ROC)
        volatility_ratio,    # Current vol / historical vol
        margin_usage,        # Margin utilization [0, 1]
        portfolio_heat,      # Overall portfolio risk
        float(is_penny_stock),  # 1.0 if price < $5
    ], dtype=np.float32)
'''
    return code


def main():
    parser = argparse.ArgumentParser(description='Export SAC policy for QuantConnect')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path')
    parser.add_argument('--print-code', action='store_true',
                        help='Print numpy inference code for QC')

    args = parser.parse_args()

    if args.print_code:
        print(create_numpy_inference_code())
        return

    export_policy_to_json(args.checkpoint, args.output)

    print("\n" + "="*60)
    print("NEXT STEPS FOR QUANTCONNECT DEPLOYMENT")
    print("="*60)
    print("1. Upload policy_export_compact.json to QC Object Store")
    print("2. In your algorithm's Initialize():")
    print("   policy_json = self.ObjectStore.Read('policy_export_compact.json')")
    print("   self.risk_policy = SACRiskPolicy(policy_dict=json.loads(policy_json))")
    print("3. In your trading logic:")
    print("   multiplier = self.risk_policy.get_position_multiplier(state)")
    print("   adjusted_quantity = base_quantity * multiplier")
    print("="*60)


if __name__ == "__main__":
    main()
