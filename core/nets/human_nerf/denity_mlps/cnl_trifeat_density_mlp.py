import torch
import torch.nn as nn


class DensityMLP(nn.Module):
    def __init__(self, input_ch=96, mlp_depth=3, mlp_width=256):
        super(DensityMLP, self).__init__()
        layers = []

        # First layer
        layers.append(nn.Linear(input_ch, mlp_width))
        layers.append(nn.LeakyReLU(negative_slope=0.01))

        # Middle layers
        for _ in range(mlp_depth - 1):
            layers.append(nn.Linear(mlp_width, mlp_width))
            layers.append(nn.LeakyReLU(negative_slope=0.01))

        # Output layer for density (sigma)
        layers.append(nn.Linear(mlp_width, 1))
        layers.append(nn.ReLU())  # Ensure non-negative density
        self.mlp = nn.Sequential(*layers)

    def forward(self, tri_feats):
        return self.mlp(tri_feats)