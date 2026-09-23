import hashlib

import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import Arm, ROW_ACTIONS
from ensomi_model.research.bounded_typed_continuation.features import CONTENT_DIM
from ensomi_model.research.bounded_typed_continuation.model import BoundedModel, ModelConfig
from ensomi_model.research.joint_audio_continuation.model import (
    JointAudioModel, JointModelConfig, initialize_from_r1,
)
from ensomi_model.research.joint_audio_continuation.state import BASE_QUERY_DIM
from ensomi_model.research.scoped_style_modeling.dataset import ContractError


def small_model(device='cpu', *, audio_levels=2):
    torch.manual_seed(171)
    return JointAudioModel(JointModelConfig(hidden=12, audio_width=8, audio_levels=audio_levels,
                                            history_levels=2, expansion=2, coupling_rank=3,
                                            routing_hidden=20, release_hidden=24)).to(device)


def legal_support(occupancy):
    return torch.tensor([[any(row) and all(a in ((0, 3) if held else (0, 1, 2))
                                          for a, held in zip(row, occupied))
                          for row in ROW_ACTIONS] for occupied in occupancy],
                        dtype=torch.bool, device=occupancy.device)


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='requires Apple MPS'))])
def test_audio_crop_halo_and_virtual_song_edges_match_full_encoding(device):
    model = small_model(device, audio_levels=6)
    halo = model.audio_halo_frames
    assert halo == 126 == JointModelConfig().audio_halo_frames
    with torch.no_grad():
        # Trained LayerNorm biases must not turn virtual padding into context.
        for block in model.audio_blocks:
            block.norm.bias.fill_(.4)
        model.set_audio_normalization(torch.linspace(-2., 2., 128), torch.linspace(.2, 3., 128))
        mel = torch.randn(1, 350, 128, device=device)
        full = model.encode_audio(mel)
        crop = model.encode_audio(mel[:, 20:330])
        torch.testing.assert_close(crop[:, halo:-halo], full[:, 20 + halo:330 - halo],
                                   atol=3e-5, rtol=3e-5)
        for start, stop in ((-halo, 20 + halo), (330 - halo, 350 + halo)):
            raw = torch.randn(1, stop - start, 128, device=device) * 20
            valid = torch.zeros(raw.shape[:2], dtype=torch.bool, device=device)
            lo, hi = max(0, start), min(350, stop)
            raw[:, lo - start:hi - start] = mel[:, lo:hi]
            valid[:, lo - start:hi - start] = True
            encoded = model.encode_audio(raw, valid)
            a, b = (0, 20) if start < 0 else (330, 350)
            torch.testing.assert_close(encoded[:, a - start:b - start], full[:, a:b],
                                       atol=3e-5, rtol=3e-5)
            assert encoded[~valid].count_nonzero() == 0


def test_audio_encoding_retains_frequency_content_and_time_resolution():
    model = small_model()
    low = torch.zeros(1, 30, 128)
    high = torch.zeros_like(low)
    low[..., 3], high[..., 100] = 1., 1.
    a, b = model.encode_audio(low), model.encode_audio(high)
    assert a.shape == (1, 30, 8)
    assert not torch.allclose(a, b)
    with pytest.raises(ContractError, match='positive scales'):
        model.set_audio_normalization(torch.zeros(128), torch.zeros(128))


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='requires Apple MPS'))])
def test_history_bos_padding_and_dense_cached_last_row_agree(device):
    model = small_model(device)
    with torch.no_grad():
        model.temporal.boundary[0].fill_(1.)
        model.temporal.boundary[1].fill_(2.)
        raw = torch.randn(4, 7, 2, CONTENT_DIM, device=device)
        valid = torch.zeros(4, 7, dtype=torch.bool, device=device)
        valid[0, :4], valid[1, 2:6] = True, True
        truncated = torch.tensor([False, True, False, True], device=device)
        encoded = model.encode_history(raw, valid, truncated)
        for batch, start in ((0, 0), (1, 2)):
            cache = model.temporal.empty_cache(truncated_start=bool(truncated[batch]))
            for row in raw[batch, start:start + 4]:
                cache = model.temporal.append(cache, row)
            torch.testing.assert_close(encoded[batch], model.temporal.read(cache), atol=2e-5, rtol=2e-5)
        assert (encoded[2] == 1.).all() and (encoded[3] == 2.).all()
        empty = model.encode_history(raw[:, :0], valid[:, :0], truncated)
        torch.testing.assert_close(empty, model.temporal.boundary[truncated.long()][:, None].expand(-1, 2, -1))
        changed = raw.clone()
        changed[~valid] *= 100
        torch.testing.assert_close(encoded, model.encode_history(changed, valid, truncated), atol=0, rtol=0)
        valid[0, 2] = False
        with pytest.raises(ContractError, match='internal'):
            model.encode_history(raw, valid, truncated)


def test_timing_and_row_distributions_are_mirror_consistent_and_chunk_independent():
    model = small_model()
    with torch.no_grad():
        torch.nn.init.normal_(model.route_residual.score[-1].weight, std=.1)
        torch.nn.init.normal_(model.release_residual.score[-1].weight, std=.1)
    audio = torch.randn(1, 9, 8)
    history = torch.randn(1, 2, 12)
    exact = torch.randn(1, 9, 2, BASE_QUERY_DIM)
    times = model.timing_logits(audio, history, exact)
    chunks = torch.cat([model.timing_logits(audio[:, :3], history, exact[:, :3]),
                        model.timing_logits(audio[:, 3:], history, exact[:, 3:])], 1)
    torch.testing.assert_close(times, chunks)
    torch.testing.assert_close(times, model.timing_logits(audio, history.flip(1), exact.flip(2)))
    occupancy = torch.tensor([[True, False, False, True]])
    legal = legal_support(occupancy)
    probs = model.row_log_probs(audio[:, 4], history, exact[:, 4], legal, occupancy)
    reverse = [ROW_ACTIONS.index(row[::-1]) for row in ROW_ACTIONS]
    mirror = model.row_log_probs(audio[:, 4], history.flip(1), exact[:, 4].flip(1),
                                 legal[:, reverse], occupancy.flip(1))
    torch.testing.assert_close(probs, mirror[:, reverse], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(probs.exp().sum(-1), torch.ones(1))
    assert torch.equal(torch.isfinite(probs), legal)


def test_head_route_never_scores_its_untrained_empty_head_mask():
    model = small_model()
    audio, history = torch.randn(1, 8), torch.randn(1, 2, 12)
    exact = torch.randn(1, 2, BASE_QUERY_DIM)
    occupancy = torch.tensor([[True, False, False, False]])
    legal = legal_support(occupancy)
    before = model.row_log_probs(audio, history, exact, legal, occupancy)
    with torch.no_grad():
        model.route_residual.score[-1].weight[0].fill_(1000.)
    after = model.row_log_probs(audio, history, exact, legal, occupancy)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    legal[:, 0] = True
    assert model.row_log_probs(audio, history, exact, legal, occupancy)[0, 0].isneginf()
    with pytest.raises(ContractError, match='no feasible'):
        model.row_log_probs(audio, history, exact, torch.zeros_like(legal), occupancy)


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='requires Apple MPS'))])
def test_both_heads_receive_audio_and_history_gradients(device):
    model = small_model(device)
    mel = torch.randn(1, 23, 128, device=device)
    raw = torch.randn(1, 5, 2, CONTENT_DIM, device=device)
    history = model.encode_history(raw, torch.ones(1, 5, dtype=torch.bool, device=device),
                                   torch.zeros(1, dtype=torch.bool, device=device))
    audio = model.encode_audio(mel)[:, 8:13]
    exact = torch.randn(1, 5, 2, BASE_QUERY_DIM, device=device)
    occupancy = torch.tensor([[True, False, False, False]], device=device)
    row = model.row_log_probs(audio[:, 2], history, exact[:, 2], legal_support(occupancy), occupancy)
    time_loss = torch.nn.functional.softplus(-model.timing_logits(audio, history, exact)).mean()
    row_loss = -row[0, ROW_ACTIONS.index((3, 2, 0, 1))]
    for loss in (time_loss, row_loss):
        for parameter in (model.audio_input.weight, model.temporal.input.weight, model.exact[0].weight):
            gradient = torch.autograd.grad(loss, parameter, retain_graph=True)[0]
            assert bool(torch.isfinite(gradient).all()) and gradient.abs().sum() > 1e-8
    (time_loss + row_loss).backward()
    assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    changed = audio.detach() + torch.randn_like(audio)
    assert not torch.allclose(model.timing_logits(audio, history, exact),
                              model.timing_logits(changed, history, exact))
    assert not torch.allclose(row, model.row_log_probs(changed[:, 2], history, exact[:, 2],
                                                      legal_support(occupancy), occupancy))


def test_r1_transfer_is_pinned_slices_historical_columns_and_records_omissions(tmp_path):
    old = BoundedModel(ModelConfig(Arm.R1, hidden=12, levels=2, expansion=2, coupling_rank=3,
                                   row_consequence='frontier2', seed_context='observed',
                                   long_memory='landmarks', memory_hidden=16,
                                   head_routing='residual', routing_hidden=20,
                                   release_routing='residual', release_hidden=24))
    checkpoint = tmp_path / 'r1.pt'
    torch.save(dict(config=dict(model=dict(arm='r1')), model=old.state_dict()), checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = small_model()
    report = initialize_from_r1(model, checkpoint, digest)
    assert report['omitted_modules'] == ['long_memory', 'row_consequence', 'seed_residual']
    assert report['parameters']['total'] == sum(p.numel() for p in model.parameters())
    assert report['discarded_exact_parameters'] > 0
    assert report['sliced'][0]['name'] == 'exact.0.weight'
    for name, source in old.state_dict().items():
        if name in report['copied']:
            expected = source[:, :BASE_QUERY_DIM] if name == 'exact.0.weight' else source
            torch.testing.assert_close(model.state_dict()[name], expected, atol=0, rtol=0)
    assert model.audio_residual.weight.count_nonzero() == 0
    audio = model.encode_audio(torch.randn(1, 12, 128))[:, 5]
    history, exact = torch.randn(1, 2, 12), torch.randn(1, 2, BASE_QUERY_DIM)
    occupancy = torch.tensor([[True, False, False, False]])
    legal = legal_support(occupancy)
    before = model.row_log_probs(audio, history, exact, legal, occupancy)
    changed_audio = audio + torch.randn_like(audio)
    torch.testing.assert_close(before, model.row_log_probs(changed_audio, history, exact, legal, occupancy),
                               atol=0, rtol=0)
    row_loss = -before[0, ROW_ACTIONS.index((3, 2, 0, 1))]
    residual_gradient = torch.autograd.grad(row_loss, model.audio_residual.weight, retain_graph=True)[0]
    assert residual_gradient.abs().sum() > 0
    time_loss = torch.nn.functional.softplus(-model.timing_logits(audio[:, None], history, exact[:, None])).mean()
    encoder_gradient = torch.autograd.grad(time_loss, model.audio_input.weight)[0]
    assert encoder_gradient.abs().sum() > 0
    with torch.no_grad():
        model.audio_residual.weight.add_(residual_gradient, alpha=-.01)
    assert not torch.allclose(model.row_log_probs(audio, history, exact, legal, occupancy),
                              model.row_log_probs(changed_audio, history, exact, legal, occupancy))
    with pytest.raises(ContractError, match='SHA-256'):
        initialize_from_r1(model, checkpoint, '0' * 64)
    incompatible = JointAudioModel(JointModelConfig(hidden=13, audio_width=8, audio_levels=2,
                                                    history_levels=2, expansion=2, coupling_rank=3))
    before = {name: value.clone() for name, value in incompatible.state_dict().items()}
    with pytest.raises(ContractError, match='shape mismatch'):
        initialize_from_r1(incompatible, checkpoint, digest)
    for name, value in incompatible.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)


def test_invalid_configurations_are_rejected():
    for kwargs in ({'hidden': True}, {'audio_levels': 11}, {'history_levels': 9}, {'audio_width': 0}):
        with pytest.raises(ContractError):
            JointModelConfig(**kwargs)
