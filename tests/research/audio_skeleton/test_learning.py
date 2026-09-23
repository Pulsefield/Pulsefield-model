import numpy as np
import pytest
import torch

from ensomi_model.research.audio_skeleton.data import event_targets, match_events, pick_events
from ensomi_model.research.audio_skeleton.model import SkeletonModel
from ensomi_model.research.audio_skeleton.training import loss_terms


def test_subframe_targets_reconstruct_events_and_reject_collisions():
    times = np.array([0., 17., 44., 99.])
    labels, offsets = event_targets(times, 11)
    positions, slots = np.nonzero(labels)
    np.testing.assert_allclose((positions + offsets[positions, slots]) * 10, times, atol=1e-5)
    labels, offsets = event_targets([10., 11.], 10)
    assert labels[1].sum() == 2
    np.testing.assert_allclose((1 + offsets[1]) * 10, [10., 11.], atol=1e-5)
    with pytest.raises(ValueError, match='capacity'):
        event_targets([10., 11., 12.], 10)
    with pytest.raises(ValueError, match='outside'):
        event_targets([1000.], 10)


def test_matching_uses_each_event_once_and_keeps_ordered_maximum():
    value = match_events([10., 20.], [0., 10.], 10.)
    assert value['matches'] == 2 and value['f1'] == 1.
    assert match_events([10., 11.], [10.], 2.)['matches'] == 1
    assert match_events([], [], 2.)['f1'] == 1.
    np.testing.assert_allclose(pick_events([.1, .8, .8, .1], np.zeros(4), .5), [10.])


def test_optional_beat_context_starts_neutral_and_learns_without_padded_loss():
    torch.manual_seed(3)
    model = SkeletonModel(use_beat_features=True, width=16)
    mel, beats, controls = torch.randn(2, 100, 128), torch.randn(2, 100, 514), torch.randn(2, 2)
    a, offset = model(mel, beats, controls)
    b, _ = model(mel, beats * 7, controls)
    torch.testing.assert_close(a, b)
    labels, targets = torch.zeros_like(a), torch.zeros_like(a)
    labels[:, 50] = 1
    mask = torch.zeros(2, 100, dtype=torch.bool); mask[:, 30:70] = True
    loss, _, _ = loss_terms(a, offset, labels, targets, mask)
    changed = labels.clone(); changed[:, :30] = 1
    altered, _, _ = loss_terms(a, offset, changed, targets, mask)
    torch.testing.assert_close(loss, altered)
    loss.backward()
    assert model.beat_projection.weight.grad.abs().sum() > 0
