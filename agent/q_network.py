"""
Convolutional Q-Network for pixel-based maze DQN (Atari-style).

Processes stacked greyscale frames of the egocentric 5×5 view rendered
at `cell_obs_px` pixels per cell (default 8 → 40×40 frames).

Input:  (batch, n_stack, H, W) float32 in [0, 1]
        Default: (batch, 4, 40, 40)

Output: (batch, n_actions) Q-values

Architecture mirrors Nature DQN (Mnih et al. 2015) with kernel sizes and
strides rescaled for the smaller 40×40 input instead of 84×84.
  Conv(4→32, 4×4, stride 2) → 32×19×19
  Conv(32→64, 3×3, stride 2) → 64×9×9
  Conv(64→64, 3×3, stride 1) → 64×7×7
  Flatten → 3136
  Linear(3136 → 512) → Linear(512 → n_actions)
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, obs_shape: tuple = (4, 40, 40), n_actions: int = 5):
        super().__init__()
        n_stack, H, W = obs_shape

        self.conv = nn.Sequential(
            nn.Conv2d(n_stack, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Compute flattened conv output size dynamically for any (H, W)
        with torch.no_grad():
            conv_out = int(self.conv(torch.zeros(1, n_stack, H, W)).numel())

        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float tensor of shape (batch, n_stack, H, W), values in [0, 1]
        Returns:
            Q-values of shape (batch, n_actions)
        """
        return self.fc(self.conv(x).flatten(1))
