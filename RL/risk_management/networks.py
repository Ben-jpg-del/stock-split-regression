"""
Neural Network Architectures for SAC Risk Management

GaussianPolicy: Actor network that outputs a Gaussian distribution over actions
QNetwork: Critic network with double Q-learning (Q1 and Q2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianPolicy(nn.Module):
    """
    SAC Actor Network - outputs Gaussian distribution over continuous actions.

    The policy outputs mean and log_std of a Gaussian distribution.
    Actions are sampled using the reparameterization trick and squashed
    through tanh to bound them to [-1, 1].

    Architecture:
        Input (state_dim) -> 256 -> 256 -> mean + log_std heads
    """

    def __init__(self, state_dim: int = 12, action_dim: int = 1, hidden_dim: int = 256):
        super().__init__()

        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Log std bounds for numerical stability
        # LOG_STD_MIN = -5 ensures std >= 0.007 (prevents entropy collapse)
        # LOG_STD_MIN = -20 was too permissive, allowed std ≈ 0 → negative entropy
        self.LOG_STD_MIN = -5
        self.LOG_STD_MAX = 2

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in self.feature:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.xavier_uniform_(self.log_std_head.weight)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(self, state: torch.Tensor) -> tuple:
        """
        Forward pass to get mean and log_std of action distribution.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            mean: Mean of Gaussian distribution (batch_size, action_dim)
            log_std: Log standard deviation (batch_size, action_dim)
        """
        features = self.feature(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)

        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple:
        """
        Sample action from the policy using reparameterization trick.

        The reparameterization trick allows gradients to flow through
        the sampling process: action = tanh(mean + std * epsilon)

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            action: Sampled action squashed to [-1, 1]
            log_prob: Log probability of the action
            mean: Mean of the distribution (for deterministic evaluation)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterized sample: z = mean + std * epsilon
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # rsample() uses reparameterization trick

        # Squash to [-1, 1] via tanh
        action = torch.tanh(x_t)

        # Compute log probability with tanh correction
        # log_prob = log_prob_normal - log(1 - tanh(x)^2)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get action from policy (convenience method).

        Args:
            state: State tensor
            deterministic: If True, return mean action (no sampling)

        Returns:
            action: Action tensor in [-1, 1]
        """
        mean, log_std = self.forward(state)

        if deterministic:
            return torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            return torch.tanh(x_t)


class QNetwork(nn.Module):
    """
    SAC Critic Network - estimates Q(s, a) using double Q-learning.

    Contains two independent Q-networks (Q1 and Q2) to reduce
    overestimation bias. The minimum of Q1 and Q2 is used for
    computing the target Q-value.

    Architecture:
        Input (state_dim + action_dim) -> 256 -> 256 -> 1 (for each Q)
    """

    def __init__(self, state_dim: int = 12, action_dim: int = 1, hidden_dim: int = 256):
        super().__init__()

        input_dim = state_dim + action_dim

        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q2 network (independent for double Q-learning)
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for network in [self.q1, self.q2]:
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple:
        """
        Compute Q-values for state-action pairs.

        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)

        Returns:
            q1: Q-value from first network
            q2: Q-value from second network
        """
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-value from Q1 network only (for actor update).

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            q1: Q-value from first network
        """
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


if __name__ == "__main__":
    # Test the networks
    print("Testing GaussianPolicy...")
    policy = GaussianPolicy(state_dim=12, action_dim=1)
    state = torch.randn(32, 12)  # Batch of 32 states

    action, log_prob, mean = policy.sample(state)
    print(f"  State shape: {state.shape}")
    print(f"  Action shape: {action.shape}")
    print(f"  Log prob shape: {log_prob.shape}")
    print(f"  Action range: [{action.min().item():.3f}, {action.max().item():.3f}]")

    print("\nTesting QNetwork...")
    critic = QNetwork(state_dim=12, action_dim=1)
    q1, q2 = critic(state, action)
    print(f"  Q1 shape: {q1.shape}")
    print(f"  Q2 shape: {q2.shape}")

    print("\n Networks initialized successfully!")
