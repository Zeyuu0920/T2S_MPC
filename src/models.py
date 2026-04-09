import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=128, num_layers=3):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
import numpy as np

def time_embedding_np(t_scalar, d=32):
    """
    phi(t) = [sin(pi/1 * t), ..., sin(pi/(d/2) * t),
              cos(pi/1 * t), ..., cos(pi/(d/2) * t)]
    return shape: (d,)
    """
    assert d % 2 == 0
    half = d // 2
    i = np.arange(1, half + 1, dtype=np.float64)
    omega = np.pi / i
    wt = omega * t_scalar
    return np.concatenate([np.sin(wt), np.cos(wt)], axis=0).astype(np.float32)


class TwoSpeedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)   # slow
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)  # slow
        self.layer3 = nn.Linear(hidden_dim, output_dim)  # fast
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.layer1(x))
        h = self.act(self.layer2(h))
        return self.layer3(h)