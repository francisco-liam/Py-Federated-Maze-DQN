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

    # Only conv layers are topology-agnostic and safe to average across clients.
    # FC layers encode Q-value magnitudes tied to each maze's BFS distances —
    # averaging them across heterogeneous topologies destroys all clients.
    _SHARED_KEY = staticmethod(lambda k: k.startswith("conv."))

    def fedavg(self, client_results, equal_weight: bool = False) -> None:
        """
        Weighted average of client weights proportional to local steps taken.
        Only convolutional layers are averaged; FC layers are left unchanged
        in the global model (partial parameter sharing).

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

        current = self._global_net.state_dict()

        if equal_weight:
            weights_list = [sd for sd, _ in client_results]
            for key in weights_list[0].keys():
                if self._SHARED_KEY(key):
                    current[key] = sum(
                        w[key].float().to(self.device) for w in weights_list
                    ) / n
        else:
            total = sum(n_steps for _, n_steps in client_results)
            if total == 0:
                return
            for key in client_results[0][0].keys():
                if self._SHARED_KEY(key):
                    current[key] = sum(
                        (n_steps / total) * weights[key].float().to(self.device)
                        for weights, n_steps in client_results
                    )
        self._global_net.load_state_dict(current)

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
        returns, ep_lengths = [], []
        successes = 0

        with torch.no_grad():
            for _ in range(n_episodes):
                ep_r, ep_len, done = 0.0, 0, False
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
                    ep_len += 1
                returns.append(ep_r)
                ep_lengths.append(ep_len)
                if ep_r > 0:
                    successes += 1

        env.close()

        return dict(
            mean_return  = float(np.mean(returns)),
            std_return   = float(np.std(returns)),
            success_rate = successes / n_episodes,
            mean_ep_len  = float(np.mean(ep_lengths)),
        )

    def transfer_evaluate(
        self,
        seed:         int,
        n_finetune:   int,
        n_eval:       int,
        env_kwargs:   dict,
        agent_kwargs: dict,
        eval_eps:     float = 0.05,
    ) -> dict:
        """
        Test whether the shared conv features transfer to a new maze.

        Creates a fresh agent, loads the global conv weights, fine-tunes
        only the FC layers for *n_finetune* episodes on *seed*, then
        evaluates for *n_eval* episodes.
        """
        from agent import DQNAgent

        agent = DQNAgent(
            obs_shape=self.obs_shape,
            n_actions=self.n_actions,
            device=str(self.device),
            **agent_kwargs,
        )
        agent.load_global_weights(self.get_weights())  # loads only conv.*

        env = MazeEnv(seed=seed, **env_kwargs)
        obs = env.reset()
        ep_return, eps_done = 0.0, 0
        while eps_done < n_finetune:
            action = agent.select_action(obs)
            next_obs, reward, done = env.step(action)
            agent.store(obs, action, reward, next_obs, done)
            agent.maybe_train()
            ep_return += reward
            obs = next_obs
            if done:
                ep_return, eps_done = 0.0, eps_done + 1

        returns, ep_lengths = [], []
        successes = 0
        with torch.no_grad():
            for _ in range(n_eval):
                obs = env.reset()
                ep_r, ep_len, done = 0.0, 0, False
                while not done:
                    if np.random.rand() < eval_eps:
                        action = np.random.randint(self.n_actions)
                    else:
                        obs_t = torch.as_tensor(
                            obs, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        action = int(agent.online_net(obs_t).argmax(dim=1).item())
                    obs, r, done = env.step(action)
                    ep_r += r
                    ep_len += 1
                returns.append(ep_r)
                ep_lengths.append(ep_len)
                if ep_r > 0:
                    successes += 1

        env.close()
        return dict(
            mean_return  = float(np.mean(returns)),
            std_return   = float(np.std(returns)),
            success_rate = successes / n_eval,
            mean_ep_len  = float(np.mean(ep_lengths)),
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
