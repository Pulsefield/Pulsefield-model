"""Audio/control mean activity for head-bearing chart rows, independent of R1."""
import math

import numpy as np
import torch

from ..typed_audio_continuation.controls import PER_FIELD_SCOPE
from ..typed_audio_continuation.demand import AudioDemand


def onset_prefix(rows):
    """Count each H row once, regardless of chord size; exclude release-only rows."""
    heads = np.isin(rows['actions'], (1, 2)).any(-1)
    return np.r_[0, heads.cumsum()]


class OnsetRate(AudioDemand):
    """The scalar mean architecture fitted to H-row counts, not head objects.

    This predictor reads audio and controls only. Its pooled audio resolution
    does not quantize event times, and one acoustic attack may support many Hs.
    """
    def __init__(self, audio_width, control_width, star_known_index, hidden=96,
                 control_encoding=PER_FIELD_SCOPE):
        super().__init__(audio_width, control_width, star_known_index, hidden, control_encoding)
        with torch.no_grad():
            self.query[-1].bias[0] = math.log(4)


def load_onset_rate(path, *, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format') != 'controlled-onset-rate/v1':
        raise ValueError('Expected a mean H-row rate checkpoint, not an object-demand model')
    model = OnsetRate(**saved['model_config'])
    model.load_state_dict(saved['model'])
    return model.to(device).eval()
