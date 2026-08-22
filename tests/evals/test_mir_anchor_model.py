from __future__ import annotations

import copy
import importlib.util

import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("requires torch", allow_module_level=True)

import torch
from torch.nn import functional as F

import pulsefield_model.evals.mir_anchor_model as mir_anchor_model_module
from pulsefield_model.evals.mir_anchor_model import ALL_FEATURE_GROUPS
from pulsefield_model.evals.mir_anchor_model import MirAnchorProbe
from pulsefield_model.evals.mir_anchor_model import MirAnchorProbeConfig
from pulsefield_model.evals.mir_anchor_model import interpolate_encoded_sequence
from pulsefield_model.evals.mir_anchor_model import triangular_support_choice_nll


def _tiny_probe(*, dropout: float = 0.0) -> MirAnchorProbe:
    return MirAnchorProbe(
        MirAnchorProbeConfig(
            history_hidden=8,
            encoder_width=8,
            embedding_dim=6,
            interaction_rank=4,
            acoustic_dilations=(1, 2),
            high_rate_dilations=(1, 2),
            tempogram_dilations=(1, 2),
            dropout=dropout,
        ),
    ).double()


def _assert_group_parameter_grads_equal(
    reference: MirAnchorProbe,
    chunked: MirAnchorProbe,
    *,
    group: str,
) -> None:
    reference_parameters = dict(reference.named_parameters())
    chunked_parameters = dict(chunked.named_parameters())
    prefix = f"feature_encoders.{group}."
    for name, parameter in reference_parameters.items():
        if not name.startswith(prefix):
            continue
        assert parameter.grad is not None, name
        assert chunked_parameters[name].grad is not None, name
        assert torch.allclose(parameter.grad, chunked_parameters[name].grad, rtol=1e-9, atol=1e-10), name


def test_probe_encodes_groups_and_scores_coalitions() -> None:
    torch.manual_seed(7)
    model = MirAnchorProbe().eval()
    batch, alternatives, support = 2, 3, 5
    history = torch.randn(batch * alternatives, 4, model.config.history_dim)
    padding = torch.tensor([[False, False, True, True]] * (batch * alternatives))
    history_state = model.encode_history(history, padding).reshape(batch, alternatives, -1)

    encoded = {
        "A": model.encode_group("A", torch.randn(1, 20, 128)),
        "N": model.encode_group("N", torch.randn(1, 20, 5)),
        "T": model.encode_group("T", torch.randn(1, 8, 122)),
        "P": model.encode_group("P", torch.randn(1, 20, 6)),
    }
    assert {name: tuple(value.shape) for name, value in encoded.items()} == {
        "A": (1, 5, 32),
        "N": (1, 20, 32),
        "T": (1, 8, 32),
        "P": (1, 20, 32),
    }

    gathered = {
        name: torch.randn(batch, alternatives, support, 32)
        for name in ("A", "N", "T", "P")
    }
    candidate_features = torch.randn(batch, alternatives, support, model.config.candidate_dim)
    baseline = model(
        history_state=history_state,
        candidate_features=candidate_features,
        embeddings={},
    )
    acoustic = model(
        history_state=history_state,
        candidate_features=candidate_features,
        embeddings=gathered,
        coalition=("A",),
    )
    full = model(
        history_state=history_state,
        candidate_features=candidate_features,
        embeddings=gathered,
        coalition=sorted(ALL_FEATURE_GROUPS),
    )

    assert baseline.shape == (batch, alternatives, support)
    assert not torch.equal(acoustic, baseline)
    assert full.shape == baseline.shape
    assert not torch.equal(full, baseline)
    assert model.parameter_count() <= model.config.max_parameters
    assert model.config.acoustic_radius_frames * 5 * model.config.acoustic_stride >= 8_000


def test_empty_history_is_zero_and_padded_values_are_ignored() -> None:
    torch.manual_seed(3)
    model = MirAnchorProbe().eval()
    history = torch.zeros(2, 3, model.config.history_dim)
    history[:, :, :2] = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
    )
    padding = torch.tensor([[False, False, True], [True, True, True]])

    state = model.encode_history(history, padding)
    changed_padding = history.clone()
    changed_padding[0, 2] = -1234.0

    assert torch.allclose(state[0], model.encode_history(changed_padding, padding)[0])
    assert torch.equal(state[1], torch.zeros_like(state[1]))


def test_fixed_shape_history_encoder_matches_packed_outputs_and_gradients() -> None:
    torch.manual_seed(23)
    fixed = _tiny_probe()
    packed_reference = copy.deepcopy(fixed)
    padding = torch.tensor(
        [
            [False, False, False, False, False],
            [False, False, False, True, True],
            [True, True, True, True, True],
        ],
    )
    fixed_history = torch.randn(3, 5, fixed.config.history_dim, dtype=torch.float64, requires_grad=True)
    packed_history = fixed_history.detach().clone().requires_grad_(True)

    actual = fixed.encode_history(fixed_history, padding)
    lengths = (~padding).sum(dim=1)
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        packed_history,
        lengths.clamp_min(1),
        batch_first=True,
        enforce_sorted=False,
    )
    _, packed_hidden = packed_reference.history_encoder(packed)
    expected = packed_hidden[-1].masked_fill((lengths == 0).unsqueeze(1), 0.0)

    assert torch.allclose(actual, expected, rtol=1e-10, atol=1e-10)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    assert fixed_history.grad is not None
    assert packed_history.grad is not None
    assert torch.allclose(fixed_history.grad, packed_history.grad, rtol=1e-9, atol=1e-10)
    fixed_parameters = dict(fixed.history_encoder.named_parameters())
    packed_parameters = dict(packed_reference.history_encoder.named_parameters())
    for name, parameter in fixed_parameters.items():
        assert parameter.grad is not None, name
        assert packed_parameters[name].grad is not None, name
        assert torch.allclose(parameter.grad, packed_parameters[name].grad, rtol=1e-9, atol=1e-10), name


def test_history_padding_must_be_trailing() -> None:
    model = MirAnchorProbe()
    history = torch.zeros(1, 3, model.config.history_dim)
    with pytest.raises(ValueError, match="padding must follow"):
        model.encode_history(history, torch.tensor([[True, False, True]]))


def test_probe_supports_history_only_and_requires_known_coalitions() -> None:
    model = MirAnchorProbe()
    history = torch.zeros(1, 64)
    candidate = torch.zeros(1, model.config.candidate_dim)
    assert model(history_state=history, candidate_features=candidate, embeddings={}).shape == (1,)
    with pytest.raises(ValueError, match="unknown audio"):
        model(
            history_state=history,
            candidate_features=candidate,
            embeddings={"A": torch.zeros(1, 32)},
            coalition=("X",),
        )


def test_encoded_sequence_interpolation_preserves_gradients_and_validity() -> None:
    sequence = torch.tensor([[0.0], [2.0], [4.0]], requires_grad=True)
    sampled, valid = interpolate_encoded_sequence(
        sequence,
        torch.tensor([0.0, 2.5, 5.0, 7.5, 12.0]),
        frame_origin_ms=0.0,
        frame_hop_ms=5.0,
        frame_valid=torch.tensor([True, True, False]),
    )

    assert torch.equal(valid, torch.tensor([True, True, True, False, False]))
    assert torch.allclose(sampled[:, 0], torch.tensor([0.0, 1.0, 2.0, 0.0, 0.0]))
    sampled.sum().backward()
    assert sequence.grad is not None
    assert torch.count_nonzero(sequence.grad) == 2


def test_encoded_sequence_interpolation_ignores_fixed_bank_tail() -> None:
    sequence = torch.tensor([[0.0], [2.0], [4.0], [100.0], [200.0]])
    sampled, valid = interpolate_encoded_sequence(
        sequence,
        torch.tensor([0.0, 5.0, 10.0, 12.5, 20.0]),
        frame_origin_ms=0.0,
        frame_hop_ms=5.0,
        frame_valid=torch.ones(3, dtype=torch.bool),
        sequence_length=3,
    )

    assert torch.equal(valid, torch.tensor([True, True, True, False, False]))
    assert torch.equal(sampled[:, 0], torch.tensor([0.0, 2.0, 4.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="sequence_length"):
        interpolate_encoded_sequence(
            sequence,
            torch.tensor([0.0]),
            frame_origin_ms=0.0,
            frame_hop_ms=5.0,
            frame_valid=torch.ones(5, dtype=torch.bool),
            sequence_length=3,
        )


def test_chunked_group_encoder_matches_outputs_and_gradients_across_seams() -> None:
    torch.manual_seed(13)
    reference = _tiny_probe()
    chunked = copy.deepcopy(reference)
    lengths = [13, 9]
    capacity = 16
    reference_features = torch.randn(2, 13, reference.config.novelty_dim, dtype=torch.float64, requires_grad=True)
    chunked_features = reference_features.detach().clone().requires_grad_(True)

    expected = [
        reference.encode_group("N", reference_features[row : row + 1, :length])[0]
        for row, length in enumerate(lengths)
    ]
    flat_bank, actual_lengths = chunked.encode_group_chunked(
        "N",
        chunked_features,
        lengths=lengths,
        chunk_size=5,
        bank_capacity=capacity,
    )
    bank = flat_bank.reshape(2, capacity, reference.config.embedding_dim)

    assert flat_bank.shape == (2 * capacity, reference.config.embedding_dim)
    assert torch.equal(actual_lengths, torch.tensor(lengths))
    for row, length in enumerate(lengths):
        assert torch.allclose(bank[row, :length], expected[row], rtol=1e-10, atol=1e-10)
        assert torch.count_nonzero(bank[row, length:]) == 0

    weights = torch.randn_like(bank)
    reference_loss = sum(
        (expected[row] * weights[row, :length]).sum()
        for row, length in enumerate(lengths)
    )
    chunked_loss = (bank * weights).sum()
    reference_loss.backward()
    chunked_loss.backward()

    assert reference_features.grad is not None
    assert chunked_features.grad is not None
    assert torch.allclose(reference_features.grad, chunked_features.grad, rtol=1e-9, atol=1e-10)
    assert torch.count_nonzero(chunked_features.grad[1, lengths[1] :]) == 0
    _assert_group_parameter_grads_equal(reference, chunked, group="N")


def test_chunked_checkpoint_preserves_dropout_outputs_and_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(29)
    checkpointed = _tiny_probe(dropout=0.25)
    direct = copy.deepcopy(checkpointed)
    features = torch.randn(2, 13, checkpointed.config.novelty_dim, dtype=torch.float64)
    direct_features = features.clone().requires_grad_(True)
    checkpointed_features = features.clone().requires_grad_(True)
    weights = torch.randn(2 * 16, checkpointed.config.embedding_dim, dtype=torch.float64)
    direct_calls = 0

    def run_direct(function, *args, **_kwargs):
        nonlocal direct_calls
        direct_calls += 1
        return function(*args)

    with monkeypatch.context() as patch:
        patch.setattr(mir_anchor_model_module, "_checkpoint", run_direct)
        torch.manual_seed(31)
        direct_bank, _ = direct.encode_group_chunked(
            "N",
            direct_features,
            lengths=[13, 9],
            chunk_size=5,
            bank_capacity=16,
        )
        (direct_bank * weights).sum().backward()

    torch.manual_seed(31)
    checkpointed_bank, _ = checkpointed.encode_group_chunked(
        "N",
        checkpointed_features,
        lengths=[13, 9],
        chunk_size=5,
        bank_capacity=16,
    )
    (checkpointed_bank * weights).sum().backward()

    assert direct_calls == 1
    assert torch.equal(checkpointed_bank, direct_bank)
    assert checkpointed_features.grad is not None
    assert direct_features.grad is not None
    assert torch.isfinite(checkpointed_features.grad).all()
    assert torch.equal(checkpointed_features.grad, direct_features.grad)
    _assert_group_parameter_grads_equal(direct, checkpointed, group="N")


def test_chunked_acoustic_encoder_matches_explicit_prepooling() -> None:
    torch.manual_seed(17)
    reference = _tiny_probe()
    chunked = copy.deepcopy(reference)
    raw = torch.randn(1, 29, reference.config.acoustic_dim, dtype=torch.float64)
    pooled = F.avg_pool1d(
        raw.transpose(1, 2),
        kernel_size=reference.config.acoustic_pool_kernel,
        stride=reference.config.acoustic_stride,
        padding=reference.config.acoustic_pool_kernel // 2,
        count_include_pad=False,
    ).transpose(1, 2)
    assert torch.allclose(
        reference.encode_group("A", raw),
        reference.encode_group("A", pooled, prepooled=True),
        rtol=1e-10,
        atol=1e-10,
    )

    reference_features = pooled.detach().clone().requires_grad_(True)
    chunked_features = pooled.detach().clone().requires_grad_(True)
    expected = reference.encode_group("A", reference_features, prepooled=True)
    flat_bank, actual_lengths = chunked.encode_group_chunked(
        "A",
        chunked_features,
        lengths=[pooled.shape[1]],
        chunk_size=3,
        bank_capacity=11,
        prepooled=True,
    )
    bank = flat_bank.reshape(1, 11, reference.config.embedding_dim)

    assert torch.equal(actual_lengths, torch.tensor([pooled.shape[1]]))
    assert torch.allclose(bank[:, : pooled.shape[1]], expected, rtol=1e-10, atol=1e-10)
    assert torch.count_nonzero(bank[:, pooled.shape[1] :]) == 0

    weights = torch.randn_like(expected)
    (expected * weights).sum().backward()
    (bank[:, : pooled.shape[1]] * weights).sum().backward()
    assert reference_features.grad is not None
    assert chunked_features.grad is not None
    assert torch.allclose(reference_features.grad, chunked_features.grad, rtol=1e-9, atol=1e-10)
    _assert_group_parameter_grads_equal(reference, chunked, group="A")


def test_chunked_group_encoder_rejects_unsafe_chunk_and_capacity() -> None:
    model = _tiny_probe()
    features = torch.randn(1, 7, model.config.novelty_dim, dtype=torch.float64)
    with pytest.raises(ValueError, match="largest block radius"):
        model.encode_group_chunked(
            "N",
            features,
            lengths=[7],
            chunk_size=4,
            bank_capacity=7,
        )
    with pytest.raises(ValueError, match="exceeds bank_capacity"):
        model.encode_group_chunked(
            "N",
            features,
            lengths=[7],
            chunk_size=5,
            bank_capacity=6,
        )
    with pytest.raises(ValueError, match="requires prepooled"):
        model.encode_group_chunked(
            "A",
            torch.randn(1, 7, model.config.acoustic_dim, dtype=torch.float64),
            lengths=[7],
            chunk_size=3,
            bank_capacity=7,
        )


def test_torch_support_loss_prefers_the_case_and_backpropagates() -> None:
    scores = torch.tensor(
        [[[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    loss = triangular_support_choice_nll(scores, half_width_ms=1)
    worse = triangular_support_choice_nll(scores.detach().flip(1), half_width_ms=1)

    assert loss < worse
    loss.backward()
    assert scores.grad is not None
