"""
DQN agent for the maze baseline.

Implements:
  - epsilon-greedy action selection with linear decay
  - experience replay via ReplayBuffer
  - separate online and target networks
  - Bellman update with smooth-L1 (Huber) loss
  - hard target network sync every `target_update_freq` gradient steps
  - gradient clipping for training stability
  - checkpoint save/load
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .q_network import QNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        obs_size: int = 26,
        n_actions: int = 5,
        hidden_size: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        target_update_freq: int = 1_000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 10_000,
        warmup_steps: int = 1_000,
        train_freq: int = 4,
        device: str = "cpu",
    ):
        """
        Args:
            obs_size:           Dimension of the flat observation vector (26).
            n_actions:          Number of discrete actions (5).
            hidden_size:        Hidden layer width for the Q-network MLP.
            lr:                 Adam learning rate.
            gamma:              Discount factor.
            batch_size:         Minibatch size for each gradient step.
            buffer_capacity:    Max transitions stored in the replay buffer.
            target_update_freq: Hard-sync target network every N gradient steps.
            eps_start:          Epsilon at step 0 (fully random).
            eps_end:            Minimum epsilon after decay.
            eps_decay_steps:    Steps over which epsilon decays linearly.
            warmup_steps:       Don't train until this many transitions are stored.
            train_freq:         Perform a gradient step every N env steps.
            device:             "cpu" or "cuda".
        """
        self.n_actions          = n_actions
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.eps_start          = eps_start
        self.eps_end            = eps_end
        self.eps_decay_steps    = eps_decay_steps
        self.warmup_steps       = warmup_steps
        self.train_freq         = train_freq
        self.device             = torch.device(device)

        self.online_net = QNetwork(obs_size, n_actions, hidden_size).to(self.device)
        self.target_net = QNetwork(obs_size, n_actions, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)

        # Counters tracked across the full training run.
        self.steps_done    = 0   # total env steps (used for epsilon and train_freq)
        self.grad_steps    = 0   # total gradient steps (used for target sync)
        self.episodes_done = 0

    # ------------------------------------------------------------------
    # Epsilon schedule
    # ------------------------------------------------------------------

    def epsilon(self) -> float:
        """Linear decay from eps_start to eps_end over eps_decay_steps."""
        frac = min(1.0, self.steps_done / max(1, self.eps_decay_steps))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> int:
        """Epsilon-greedy action selection for training."""
        if np.random.rand() < self.epsilon():
            return np.random.randint(self.n_actions)
        return self._greedy_action(obs)

    def select_greedy_action(self, obs: np.ndarray) -> int:
        """Pure greedy — used during evaluation only."""
        return self._greedy_action(obs)

    def _greedy_action(self, obs: np.ndarray) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.online_net(obs_t).argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition and increment the step counter."""
        self.buffer.push(state, action, reward, next_state, done)
        self.steps_done += 1

    def maybe_train(self) -> Optional[float]:
        """
        Perform a gradient step if conditions are met.

        Conditions:
          1. Replay buffer has at least warmup_steps transitions.
          2. steps_done is a multiple of train_freq.

        Returns the loss value if a step was taken, else None.
        """
        if len(self.buffer) < self.warmup_steps:
            return None
        if self.steps_done % self.train_freq != 0:
            return None
        return self._gradient_step()

    def _gradient_step(self) -> float:
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t      = torch.as_tensor(states,      dtype=torch.float32, device=self.device)
        actions_t     = torch.as_tensor(actions,     dtype=torch.int64,   device=self.device).unsqueeze(1)
        rewards_t     = torch.as_tensor(rewards,     dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t       = torch.as_tensor(dones,       dtype=torch.float32, device=self.device)

        # Q(s, a) for the actions actually taken.
        q_values = self.online_net(states_t).gather(1, actions_t).squeeze(1)

        # Bellman target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        # done=1.0 masks out the bootstrap term for terminal transitions.
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1).values
            targets    = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients in early training.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.grad_steps += 1

        # Hard target network sync.
        if self.grad_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "online_net":    self.online_net.state_dict(),
            "target_net":    self.target_net.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "steps_done":    self.steps_done,
            "grad_steps":    self.grad_steps,
            "episodes_done": self.episodes_done,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done    = ckpt.get("steps_done",    0)
        self.grad_steps    = ckpt.get("grad_steps",    0)
        self.episodes_done = ckpt.get("episodes_done", 0)
