"""Shared audio-origin cues for active holds, without predicted endpoints."""
import torch
from torch import nn

from ..bounded_typed_continuation.features import RELATIVE_LANES
from ..scoped_style_modeling.dataset import ContractError


class HoldAudioCues(nn.Module):
    def __init__(self, audio_width, hidden, row_width):
        super().__init__()
        self.audio_width = audio_width
        self.slots = nn.Sequential(nn.Linear(2*audio_width+1, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU())
        self.release = nn.Linear(hidden, row_width, bias=False)
        self.row = nn.Linear(4*hidden, row_width, bias=False)
        nn.init.zeros_(self.release.weight)
        nn.init.zeros_(self.row.weight)
        self.register_buffer('relative_lanes', torch.tensor(RELATIVE_LANES), persistent=False)

    def encode(self, audio, holds):
        """Read [origin audio, asinh age in seconds, active bit] for four slots."""
        if holds is None or holds.shape != (len(audio), 4, self.audio_width+2):
            raise ContractError('Active-LN audio cues require four aligned origin/age/occupancy slots')
        current = audio[:, None].expand(-1, 4, -1)
        return self.slots(torch.cat((holds[..., :-1], current), -1))*holds[..., -1:]

    def release_values(self, audio, holds):
        # Pooling offers a release-time preference; R1 selects the release subset.
        return self.release(self.encode(audio, holds).sum(-2))

    def row_values(self, audio, holds):
        slots = self.encode(audio, holds)
        return self.row(slots[:, self.relative_lanes].flatten(-2))
