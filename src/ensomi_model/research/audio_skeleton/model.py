"""Local timing detail and optional frozen musical context on a 10 ms clock."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width, dilation, kernel=5):
        super().__init__()
        self.conv = nn.Conv1d(width, width, kernel, padding=dilation * (kernel // 2),
                              dilation=dilation, groups=width)
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, value):
        mixed = self.conv(value.transpose(1, 2)).transpose(1, 2)
        return value + self.ff(self.norm(mixed))


class SkeletonModel(nn.Module):
    def __init__(self, *, use_beat_features=False, beat_width=514, width=128):
        super().__init__()
        self.use_beat_features = use_beat_features
        self.beat_width, self.width = beat_width, width
        self.local = nn.Sequential(nn.Linear(128, width), nn.GELU(), nn.LayerNorm(width))
        self.condition = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))
        self.detail = nn.Sequential(*(Block(width, d) for d in (1, 2, 4, 8, 16, 32)))
        self.phrase = nn.Sequential(*(Block(width, d, 3) for d in (1, 2, 4, 8)))
        self.norm = nn.LayerNorm(width)
        self.events = nn.Linear(width, 2)
        self.offsets = nn.Linear(width, 2)
        # Both arms instantiate the same modules, preserving shared initialization.
        self.beat_norm = nn.LayerNorm(beat_width)
        self.beat_projection = nn.Linear(beat_width, width, bias=False)
        nn.init.zeros_(self.beat_projection.weight)

    def forward(self, mel, beats, controls):
        value = self.local((mel + 4.) / 4.) + self.condition(controls)[:, None]
        if self.use_beat_features:
            value = value + self.beat_projection(self.beat_norm(beats))
        slow = F.avg_pool1d(value.transpose(1, 2), 10, stride=10)
        slow = self.phrase(slow.transpose(1, 2)).transpose(1, 2)
        slow = F.interpolate(slow, size=value.shape[1], mode='linear', align_corners=False).transpose(1, 2)
        hidden = self.norm(self.detail(value) + slow)
        return self.events(hidden), self.offsets(hidden).tanh() * .5
