"""Fully connected PINN: (x*, y*, s) -> (u*, v*, p*)."""

import math

import torch
import torch.nn as nn


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class PINN(nn.Module):
    def __init__(self, lower, upper, width=128, depth=5, activation="tanh",
                 omega0=30.0, fourier_features=0, fourier_sigma=1.0, seed=0):
        super().__init__()
        # Affine map of the physical input box to [-1, 1]^3 (stored with the model).
        self.register_buffer("lower", torch.as_tensor(lower, dtype=torch.float32))
        self.register_buffer("upper", torch.as_tensor(upper, dtype=torch.float32))
        self.activation, self.omega0 = activation, omega0

        gen = torch.Generator().manual_seed(seed)
        n_in = 3
        if fourier_features > 0:
            # Random Fourier embedding of the spatial coordinates only; s passes through.
            self.register_buffer("B", fourier_sigma * torch.randn(2, fourier_features, generator=gen))
            n_in = 2 * fourier_features + 1
        else:
            self.B = None

        dims = [n_in] + [width] * depth + [3]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.act = Sine() if activation == "sine" else nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        for i, lin in enumerate(self.layers):
            if self.activation == "sine":   # SIREN initialisation (Sitzmann et al. 2020)
                bound = 1.0 / lin.in_features if i == 0 else math.sqrt(6.0 / lin.in_features) / self.omega0
                nn.init.uniform_(lin.weight, -bound, bound)
            else:
                nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, xys):
        h = 2.0 * (xys - self.lower) / (self.upper - self.lower) - 1.0
        if self.B is not None:
            proj = 2.0 * math.pi * h[:, :2] @ self.B
            h = torch.cat([torch.sin(proj), torch.cos(proj), h[:, 2:3]], dim=1)
        for i, lin in enumerate(self.layers[:-1]):
            h = lin(h)
            h = self.act(self.omega0 * h if (self.activation == "sine" and i == 0) else h)
        return self.layers[-1](h)
