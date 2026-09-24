"""Legacy source-query diagnostics with the full-song audio information contract."""
from .batching import JointScores, interpolate_audio


def score_context_queries(model, inputs, coarse=None, *, code=None):
    """Score aligned queries using externally encoded complete-song context.

Coarse tensors must have the same batch/song order as inputs. Reusing their
gradients is permitted, but a crop cannot substitute for full audio context.
"""
    encoded = model.encode_crop(inputs.mel, inputs.mel_valid, inputs.mel_starts, inputs.mel_frame_counts, coarse)
    from .intent_model import IntentAudioModel
    if isinstance(model, IntentAudioModel):
        encoded = model.apply_code(encoded, code)
    elif code is not None:
        raise ValueError('This context-query model has no intent code')
    history = model.encode_history(inputs.raw, inputs.history_valid, inputs.truncated)
    timing_audio = interpolate_audio(encoded, inputs.timing_times_ms, inputs.mel_starts, inputs.mel_frame_counts)
    row_audio = interpolate_audio(encoded, inputs.row_times_ms, inputs.mel_starts, inputs.mel_frame_counts)
    timing = model.timing_logits(timing_audio, history, inputs.timing_exact).flatten(1)
    rows = model.row_log_probs(row_audio, history, inputs.row_exact, inputs.row_legal, inputs.occupancy)
    return JointScores(timing, rows)
