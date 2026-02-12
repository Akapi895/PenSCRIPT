"""
Simple DQN Agent for Curriculum Testing — Phase 3.

A lightweight agent that works DIRECTLY with PenGym/NASim environments.
This is NOT SCRIPT's PPO — it's an independent DQN for validating the
curriculum learning mechanism.

Design rationale:
  - Uses gymnasium API directly → no SCRIPT dependency
  - Flat observation (NASim obs) → no SBERT encoding needed
  - Simple epsilon-greedy DQN → fast training for comparison tests
  - Same interface for both flat and curriculum controllers
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from typing import Optional, Tuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """Simple experience replay buffer."""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Simple MLP Q-network."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class SimpleDQNAgent:
    """Lightweight DQN agent for curriculum testing.

    Directly interacts with PenGym/NASim gymnasium environments.
    No SBERT, no SCRIPT dependencies — purely for validating the
    curriculum vs flat training comparison.
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: int = 5000,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 target_update: int = 100,
                 hidden_dim: int = 128,
                 device: str = 'auto',
                 seed: int = 42):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update

        # Device
        if device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Seed
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Networks
        self.policy_net = QNetwork(
            obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(
            obs_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self.steps_done = 0
        self.updates_done = 0

    @property
    def epsilon(self) -> float:
        """Current epsilon for epsilon-greedy."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-self.steps_done / self.epsilon_decay)

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        """Select action using epsilon-greedy.

        Args:
            obs: Observation from the environment
            eval_mode: If True, use greedy (no exploration)

        Returns:
            Action index
        """
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax(dim=1).item()

    def store_transition(self, obs, action, reward, next_obs, done):
        """Store a transition in replay buffer."""
        self.buffer.push(obs, action, reward, next_obs, done)
        self.steps_done += 1

    def update(self) -> Optional[float]:
        """Run one gradient step.

        Returns:
            Loss value or None if buffer too small
        """
        if len(self.buffer) < self.batch_size:
            return None

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(
            np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        # Q(s, a)
        q_values = self.policy_net(states).gather(1, actions)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.functional.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.updates_done += 1

        # Target network update
        if self.updates_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str):
        """Save model weights."""
        from pathlib import Path as P
        P(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'updates_done': self.updates_done,
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device,
                                weights_only=True)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint.get('steps_done', 0)
        self.updates_done = checkpoint.get('updates_done', 0)

    def get_stats(self) -> dict:
        """Return agent statistics."""
        return {
            'steps_done': self.steps_done,
            'updates_done': self.updates_done,
            'epsilon': self.epsilon,
            'buffer_size': len(self.buffer),
        }
