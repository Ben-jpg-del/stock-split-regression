"""
Experience Replay Buffer for SAC

Stores transitions (state, action, reward, next_state, done) and
provides efficient batch sampling for training.
"""

import numpy as np
import torch
from collections import deque, namedtuple
import random
from typing import Tuple, Optional


# Named tuple for storing transitions
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Experience replay buffer with FIFO eviction.

    Uses a deque with maxlen for automatic oldest-experience eviction.
    Samples are converted to numpy arrays first, then to tensors for efficiency.
    """

    def __init__(self, capacity: int = 1_000_000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        # Ensure proper types
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)

        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample
            device: Device to move tensors to ('cpu' or 'cuda')

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
            Note: weights are uniform (ones) for standard replay buffer.
                  indices are sample positions (for API compatibility with PrioritizedReplayBuffer).
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        experiences = [self.buffer[i] for i in indices]

        # Convert to numpy arrays first (much faster than direct tensor conversion)
        states = np.array([e.state for e in experiences], dtype=np.float32)
        actions = np.array([e.action for e in experiences], dtype=np.float32)
        rewards = np.array([e.reward for e in experiences], dtype=np.float32)
        next_states = np.array([e.next_state for e in experiences], dtype=np.float32)
        dones = np.array([e.done for e in experiences], dtype=np.float32)

        # Convert to tensors and move to device
        states = torch.from_numpy(states).to(device)
        actions = torch.from_numpy(actions).to(device)
        rewards = torch.from_numpy(rewards).unsqueeze(1).to(device)
        next_states = torch.from_numpy(next_states).to(device)
        dones = torch.from_numpy(dones).unsqueeze(1).to(device)

        # Uniform weights for API compatibility with PrioritizedReplayBuffer
        weights = torch.ones(batch_size, 1, device=device)

        return states, actions, rewards, next_states, dones, weights, indices

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return len(self.buffer) >= batch_size

    def clear(self):
        """Clear all experiences from buffer."""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer (optional, for improved sample efficiency).

    Samples transitions with probability proportional to their TD error,
    focusing learning on more "surprising" transitions.
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames to anneal beta to 1.0
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition with maximum priority."""
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0

        experience = Experience(
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            reward,
            np.asarray(next_state, dtype=np.float32),
            done
        )

        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, device: str = 'cpu') -> Tuple[torch.Tensor, ...]:
        """Sample batch with priorities."""
        # Compute sampling probabilities over available experiences
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs /= prob_sum
        else:
            # Fallback to uniform if all priorities are zero
            probs = np.ones(self.size, dtype=np.float32) / self.size

        # Use replacement when buffer is smaller than batch size
        use_replacement = self.size < batch_size
        indices = np.random.choice(self.size, batch_size, p=probs, replace=use_replacement)

        # Compute importance sampling weights
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.from_numpy(weights.astype(np.float32)).unsqueeze(1).to(device)

        # Get experiences
        experiences = [self.buffer[i] for i in indices]

        states = torch.from_numpy(np.array([e.state for e in experiences])).to(device)
        actions = torch.from_numpy(np.array([e.action for e in experiences])).to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences])).unsqueeze(1).to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences])).to(device)
        dones = torch.from_numpy(np.array([e.done for e in experiences])).unsqueeze(1).to(device)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small epsilon for stability

    def __len__(self) -> int:
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size


if __name__ == "__main__":
    # Test the replay buffer
    print("Testing ReplayBuffer...")
    buffer = ReplayBuffer(capacity=10000)

    # Add some experiences
    for i in range(1000):
        state = np.random.randn(12).astype(np.float32)
        action = np.random.randn(1).astype(np.float32)
        reward = np.random.randn()
        next_state = np.random.randn(12).astype(np.float32)
        done = i % 100 == 0

        buffer.push(state, action, reward, next_state, done)

    print(f"  Buffer size: {len(buffer)}")
    print(f"  Is ready for batch of 256: {buffer.is_ready(256)}")

    # Sample a batch
    states, actions, rewards, next_states, dones, weights, indices = buffer.sample(256)
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Next states shape: {next_states.shape}")
    print(f"  Dones shape: {dones.shape}")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Indices shape: {indices.shape}")

    print("\n Replay buffer working correctly!")
