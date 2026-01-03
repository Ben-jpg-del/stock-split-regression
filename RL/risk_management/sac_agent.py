import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional, Tuple
import os

from .networks import GaussianPolicy, QNetwork
from .replay_buffer import ReplayBuffer


class SACAgent:

    def __init__(
        self,
        state_dim: int = 11,
        action_dim: int = 1,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-5,  # 10x slower than actor/critic for stability
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        target_entropy: Optional[float] = None,
        device: str = 'auto'
    ):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.lr_alpha = lr_alpha

        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        weight_decay = 1e-5
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=weight_decay)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)

        self.target_entropy = target_entropy if target_entropy is not None else -action_dim
        # Initialize alpha to 0.2 (log(0.2) ≈ -1.6) for reasonable exploration
        self.log_alpha = torch.tensor([-1.6], requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr_alpha)
        self.min_alpha = 0.01  # Minimum alpha to prevent collapse

        self.buffer = ReplayBuffer(buffer_size)

        self.training_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(
        self,
        state: np.ndarray,
        evaluate: bool = False
    ) -> np.ndarray:
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.actor.sample(state)

        return action.cpu().numpy()[0]

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float]:
        if not self.buffer.is_ready(self.batch_size):
            return {}

        self.training_steps += 1

        states, actions, rewards, next_states, dones, _weights, _indices = self.buffer.sample(
            self.batch_size, self.device
        )

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)

            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_probs

            q_target = rewards + self.gamma * (1 - dones) * q_next

        q1, q2 = self.critic(states, actions)

        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optim.step()

        new_actions, log_probs, _ = self.actor.sample(states)

        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # Clamp alpha to prevent collapse (minimum exploration)
        with torch.no_grad():
            self.log_alpha.clamp_(min=np.log(self.min_alpha))

        self._soft_update_target()

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'entropy': -log_probs.mean().item(),
            'q_value': q_new.mean().item()
        }

    def _soft_update_target(self):
        for param, target_param in zip(
            self.critic.parameters(),
            self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path: str):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'alpha_optim_state_dict': self.alpha_optim.state_dict(),
            'training_steps': self.training_steps,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'tau': self.tau,
                'target_entropy': self.target_entropy,
                'lr_alpha': self.lr_alpha,
            }
        }
        torch.save(checkpoint, path)
        print(f"Agent saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.log_alpha = checkpoint['log_alpha'].to(self.device).requires_grad_(True)
        config = checkpoint.get('config', {})
        lr_alpha = config.get('lr_alpha', self.lr_alpha)
        self.lr_alpha = lr_alpha
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr_alpha)
        self.alpha_optim.load_state_dict(checkpoint['alpha_optim_state_dict'])
        self.training_steps = checkpoint['training_steps']

        print(f"Agent loaded from {path}")
        print(f"  Training steps: {self.training_steps}")
        print(f"  Alpha: {self.alpha.item():.4f}")

    def action_to_multiplier(self, action: float) -> float:
        return (action + 1) / 2


def create_sac_agent(
    state_dim: int = 11,
    action_dim: int = 1,
    device: str = 'auto',
    **kwargs
) -> SACAgent:
    default_kwargs = {
        'hidden_dim': 256,
        'gamma': 0.99,
        'tau': 0.005,
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'lr_alpha': 3e-5,  # 10x slower for stability
        'buffer_size': 1_000_000,
        'batch_size': 256,
    }
    default_kwargs.update(kwargs)

    return SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        **default_kwargs
    )


if __name__ == "__main__":
#Test the SACAgent
    print("Testing SACAgent...")

    agent = create_sac_agent(state_dim=11, action_dim=1)
    print(f"  Device: {agent.device}")
    print(f"  Initial alpha: {agent.alpha.item():.4f}")

    state = np.random.randn(11).astype(np.float32)
    action = agent.select_action(state)
    print(f"  Sample action: {action}")
    print(f"  Position multiplier: {agent.action_to_multiplier(action[0]):.3f}")

    for i in range(500):
        state = np.random.randn(11).astype(np.float32)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(11).astype(np.float32)
        done = i % 100 == 0

        agent.store_transition(state, action, reward, next_state, done)

    print(f"  Buffer size: {len(agent.buffer)}")

    # Perform some updates
    print("\n  Performing training updates...")
    for i in range(10):
        metrics = agent.update()
        if metrics:
            print(f"    Step {i+1}: critic_loss={metrics['critic_loss']:.4f}, "
                  f"alpha={metrics['alpha']:.4f}")

    print("\n SAC Agent working correctly!")
