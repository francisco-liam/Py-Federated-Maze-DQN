"""
Uniform experience replay buffer for the DQN baseline.

Stores (state, action, reward, next_state, done) tuples in a fixed-size
circular buffer. Sampling is uniform random (no prioritization).
Prioritized replay is a later optimization once the baseline is working.
"""

import random
from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of transitions to store.
                      Oldest transitions are evicted when full.
        """
        self._buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random minibatch.

        Returns
        -------
        states      : (batch, obs_size)  float32
        actions     : (batch,)           int64
        rewards     : (batch,)           float32
        next_states : (batch, obs_size)  float32
        dones       : (batch,)           float32  (1.0 if terminal, else 0.0)
        """
        batch = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.stack(next_states),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)
