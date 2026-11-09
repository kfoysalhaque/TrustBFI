# model.py
# Simple encoder + projector + classifier for V-matrix fingerprinting
# Input x: (B, 16, 250)

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x):
        # x: (B, 16, 250)
        h = self.net(x).squeeze(-1)   # (B, 128)
        z = self.fc(h)                # (B, emb_dim)
        return z


class Projector(nn.Module):
    def __init__(self, emb_dim=128, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, proj_dim),
        )

    def forward(self, z):
        return self.net(z)


class Classifier(nn.Module):
    def __init__(self, emb_dim=128, num_classes=15):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, z):
        return self.fc(z)
