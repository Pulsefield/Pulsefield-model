"""R1 count-family baseline from audio, timing, holds and scoped controls."""
import torch
from torch import nn

from ..bounded_typed_continuation.features import TIME_DIM
from ..bounded_typed_continuation.temporal import pointwise


class CompositionPrior(nn.Module):
    """Predict (head count, release count) logits without row-content memory.

    LN ages include their availability bits. Reading both relative hand views
    and averaging their logits preserves mirror symmetry. Exact row support
    and the consequence path still receive the full committed state.
    """
    def __init__(self, audio_width, preview_width, control_width, groups, hidden=128):
        super().__init__()
        self.readout = nn.Sequential(nn.Linear(audio_width+preview_width+control_width+4*TIME_DIM, hidden),
                                     nn.GELU(), nn.Linear(hidden, groups))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        # exact_features stores (LN age, last attack, last release) per lane.
        positions = [lane*3*TIME_DIM+i for lane in range(4) for i in range(TIME_DIM)]
        self.register_buffer('hold_positions', torch.tensor(positions), persistent=False)

    def forward(self, audio, exact, preview, control):
        common = torch.cat((audio, preview, control), -1).unsqueeze(-2).expand(-1, 2, -1)
        holds = exact.index_select(-1, self.hold_positions)
        return pointwise(self.readout, torch.cat((common, holds), -1)).mean(-2)
