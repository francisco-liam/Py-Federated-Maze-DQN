"""
FL server: holds the global QNetwork and performs FedAvg aggregation.
Also evaluates the global model on an arbitrary seed.
"""

import numpy as np
import torch

from agent.q_network import QNetwork
from maze_env import MazeEnv


class FLServer:
    def __init__(self, obs_shape: tuple, n_actions: int, device: str) -> None:
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.device    = torch.device(device)

        self._global_net = QNetwork(obs_shape, n_actions).to(self.device)
        # No gradients needed on the server — updates come from aggregation only
        for p in self._global_net.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Weight access
    # ------------------------------------------------------------------

    def get_weights(self) -> dict:
        """Return a detached copy of the current global weights."""
        return {k: v.clone() for k, v in self._global_net.state_dict().items()}

    # ------------------------------------------------------------------
    # FedAvg aggregation
    # ------------------------------------------------------------------

    def fedavg(self, client_results, equal_weight: bool = False) -> None:
        """
        Weighted average of client weights proportional to local steps taken.

        Parameters
        ----------
        client_results : list of (state_dict, n_steps)
        equal_weight   : if True, average uniformly (ignores n_steps).
                         Useful when clients learn at very different speeds —
                         step-weighted FedAvg can be dominated by the fastest
                         learner even at low weight because its signal is less noisy.
        """
        n = len(client_results)
        if n == 0:
            return

        if equal_weight:
            weights_list = [sd for sd, _ in client_results]
            new_weights  = {}
            for key in weights_list[0].keys():
                new_weights[key] = sum(
                    w[key].float().to(self.device) for w in weights_list
                ) / n
        else:
            total = sum(n_steps for _, n_steps in client_results)
            if total == 0:
                return
            new_weights = {}
            for key in client_results[0][0].keys():
                new_weights[key] = sum(
                    (n_steps / total) * weights[key].float().to(self.device)
                    for weights, n_steps in client_results
                )
        self._global_net.load_state_dict(new_weights)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        seed:        int,
        n_episodes:  int,
        env_kwargs:  dict,
        eval_eps:    float = 0.05,
    ) -> dict:
        """
        Evaluate the global model on *seed* for *n_episodes* episodes.
        Uses near-greedy policy (eval_eps=0.05) matching training protocol.
        """
        net = self._global_net
        net.eval()

        env     = MazeEnv(seed=seed, **env_kwargs)
        obs     = env.reset()
        returns = []
        successes = 0

        with torch.no_grad():
            for _ in range(n_episodes):
                ep_r, done = 0.0, False
                while not done:
                    if np.random.rand() < eval_eps:
                        action = np.random.randint(self.n_actions)
                    else:
                        obs_t  = torch.as_tensor(
                            obs, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        action = int(net(obs_t).argmax(dim=1).item())
                    obs, r, done = env.step(action)
                    ep_r += r
                returns.append(ep_r)
                if ep_r > 0:
                    successes += 1

        env.close()

        return dict(
            mean_return  = float(np.mean(returns)),
            std_return   = float(np.std(returns)),
            success_rate = successes / n_episodes,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(self._global_net.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        # Support both bare QNetwork state_dicts and full agent checkpoints
        # (the baseline is saved by DQNAgent.save() which wraps the nets in a dict)
        if isinstance(state, dict) and "online_net" in state:
            state = state["online_net"]
        self._global_net.load_state_dict(state)
