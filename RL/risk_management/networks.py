import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int = 11, action_dim: int = 1, hidden_dim: int = 256):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        self.LOG_STD_MIN = -2  # exp(-2) ≈ 0.14, prevents over-deterministic policy
        self.LOG_STD_MAX = 2

        self._init_weights()

    def _init_weights(self):
        for layer in self.feature:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.xavier_uniform_(self.log_std_head.weight)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(self, state: torch.Tensor) -> tuple:
        features = self.feature(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)

        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple:
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() 

        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self.forward(state)

        if deterministic:
            return torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            return torch.tanh(x_t)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = 11, action_dim: int = 1, hidden_dim: int = 256):
        super().__init__()

        input_dim = state_dim + action_dim

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for network in [self.q1, self.q2]:
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


if __name__ == "__main__":
    # Test the networks
    print("Testing GaussianPolicy...")
    policy = GaussianPolicy(state_dim=11, action_dim=1)
    state = torch.randn(32, 11)  # Batch of 32 states

    action, log_prob, mean = policy.sample(state)
    print(f"  State shape: {state.shape}")
    print(f"  Action shape: {action.shape}")
    print(f"  Log prob shape: {log_prob.shape}")
    print(f"  Action range: [{action.min().item():.3f}, {action.max().item():.3f}]")

    print("\nTesting QNetwork...")
    critic = QNetwork(state_dim=11, action_dim=1)
    q1, q2 = critic(state, action)
    print(f"  Q1 shape: {q1.shape}")
    print(f"  Q2 shape: {q2.shape}")

    print("\n Networks initialized successfully!")
