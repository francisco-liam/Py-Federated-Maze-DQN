"""
FL client: owns one MazeEnv (fixed seed) + one DQNAgent.

Each round:
  1. Receive global weights from server.
  2. Load them into the agent (reset optimizer, keep replay buffer + counters).
  3. Optionally set FedProx proximal term.
  4. Run `n_episodes` of local DQN training.
  5. Return (local_weights, steps_taken, per_episode_returns).
"""

from typing import Optional

import numpy as np
import torch

from agent import DQNAgent
from maze_env import MazeEnv


class FLClient:
    def __init__(
        self,
        client_id:    int,
        seed:         int,
        obs_shape:    tuple,
        n_actions:    int,
        device:       str,
        agent_kwargs: dict,
        env_kwargs:   dict,
    ) -> None:
        self.client_id = client_id
        self.seed      = seed

        self.env   = MazeEnv(seed=seed, **env_kwargs)
        self.agent = DQNAgent(
            obs_shape=obs_shape,
            n_actions=n_actions,
            device=device,
            **agent_kwargs,
        )
        self._obs = self.env.reset()

    # ------------------------------------------------------------------
    # Per-round training
    # ------------------------------------------------------------------

    def train_round(
        self,
        global_weights: dict,
        n_episodes:     int,
        proximal_mu:    float = 0.0,
    ):
        """
        Load global weights and train locally for *n_episodes* episodes.

        Returns
        -------
        local_weights : dict   online_net state_dict after local training
        steps_taken   : int    env steps taken this round
        ep_returns    : list   per-episode undiscounted returns
        """
        self.agent.load_global_weights(global_weights)

        if proximal_mu > 0.0:
            self.agent.set_proximal_term(global_weights, proximal_mu)
        else:
            self.agent.clear_proximal_term()

        steps_start = self.agent.steps_done
        ep_returns  = []
        ep_return   = 0.0
        eps_done    = 0

        while eps_done < n_episodes:
            action              = self.agent.select_action(self._obs)
            next_obs, reward, done = self.env.step(action)
            self.agent.store(self._obs, action, reward, next_obs, done)
            self.agent.maybe_train()
            ep_return += reward
            self._obs  = next_obs

            if done:
                ep_returns.append(ep_return)
                print(
                    f"  [client {self.client_id}|seed {self.seed}] "
                    f"ep {eps_done+1}/{n_episodes}  "
                    f"return={ep_return:+.2f}  "
                    f"eps={self.agent.epsilon():.3f}",
                    flush=True,
                )
                ep_return = 0.0
                eps_done += 1

        return (
            self.agent.get_online_weights(),
            self.agent.steps_done - steps_start,
            ep_returns,
        )

    def local_evaluate(self, n_episodes: int, eval_eps: float = 0.05) -> dict:
        """
        Evaluate the client's current local model (shared conv + local FC)
        on its own maze. Resets the env state for the next train_round.
        """
        returns = []
        successes = 0
        with torch.no_grad():
            for _ in range(n_episodes):
                obs = self.env.reset()
                ep_r, done = 0.0, False
                while not done:
                    if np.random.rand() < eval_eps:
                        action = np.random.randint(self.agent.n_actions)
                    else:
                        obs_t = torch.as_tensor(
                            obs, dtype=torch.float32, device=self.agent.device
                        ).unsqueeze(0)
                        action = int(self.agent.online_net(obs_t).argmax(dim=1).item())
                    obs, reward, done = self.env.step(action)
                    ep_r += reward
                returns.append(ep_r)
                if ep_r > 0:
                    successes += 1
        self._obs = self.env.reset()
        return dict(
            mean_return  = float(np.mean(returns)),
            success_rate = successes / n_episodes,
        )

    def close(self) -> None:
        self.env.close()
