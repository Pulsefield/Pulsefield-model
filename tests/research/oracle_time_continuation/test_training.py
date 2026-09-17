from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from pulsefield_model.research.oracle_time_continuation.config import BackboneConfig
from pulsefield_model.research.oracle_time_continuation.engine import ContinuationEngine
from pulsefield_model.research.oracle_time_continuation.model import CausalBackbone, row_index
from pulsefield_model.research.oracle_time_continuation.objective import MARGINAL_NAMES, ObjectiveConfig
from pulsefield_model.research.oracle_time_continuation.training import SequenceTrainer
from pulsefield_model.research.oracle_time_continuation.training_config import TrainingConfig
from pulsefield_model.research.oracle_time_continuation.windows import WindowSampler
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError
from .conftest import admit
from .test_objective import structural_values
from .test_windows import tap_source


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_model():
    torch.manual_seed(46)
    return CausalBackbone(BackboneConfig(hidden=8, heads=2, recent=4, coarse_group=2, coarse_capacity=2,
                                       coupling_rank=4, attacks_per_lane=2, releases_per_lane=1))


def ragged_windows():
    sources = (tap_source(70), tap_source(48, group="song:b"))
    sampler = WindowSampler(sources)
    return (sampler.window(sources[0].identity.source_sha256, 30, 0),
            sampler.window(sources[1].identity.source_sha256, 41, 0),
            sampler.window(sources[0].identity.source_sha256, 68, 2))


def reference_costs(model, windows):
    engine = ContinuationEngine(model)
    nll, marginal = 0., torch.zeros(3)
    with torch.no_grad():
        for window in windows:
            state = engine.prefill(window.source.skeleton, window.source.targets[:window.start])
            for row in window.source.targets[window.start:window.stop]:
                distribution = engine.predict(state)
                nll -= float(distribution.score(row.actions))
                support = state.execution.query().legal_actions
                occupied = state.execution.replay.occupancy
                truth = structural_values(row.actions, occupied)
                for group in range(3):
                    indices = [row_index(actions) for actions in support
                               if structural_values(actions, occupied)[group] == truth[group]]
                    marginal[group] -= distribution.log_probs[indices].logsumexp(0)
                state = engine.commit(state, row)
    return nll, marginal


@pytest.mark.parametrize("coefficient", (0., .6))
def test_complete_effective_batch_denominator_microbatch_parity_and_single_clip_step(monkeypatch, coefficient):
    model = small_model()
    windows = ragged_windows()
    expected_nll, expected_marginal = reference_costs(model, windows)
    training = TrainingConfig(effective_batch_size=3, microbatch_size=2, chunk_rows=4, max_grad_norm=1e6)
    objective = ObjectiveConfig(lambda_struct=coefficient)
    first = SequenceTrainer(deepcopy(model), training, objective)
    second = SequenceTrainer(deepcopy(model), replace(training, microbatch_size=1), objective)
    supervised, events = 0, []
    real_batch = first.engine.teacher_force_batch
    real_clip = torch.nn.utils.clip_grad_norm_
    real_step = first.optimizer.step

    def batch(states, chunks):
        nonlocal supervised
        assert all(not value.requires_grad for state in states for token in state.temporal.tokens for value in token.inputs)
        supervised += sum(map(len, chunks))
        events.append("chunk")
        return real_batch(states, chunks)

    def clip(*args, **kwargs):
        assert supervised == sum(window.target_rows for window in windows)
        events.append("clip")
        return real_clip(*args, **kwargs)

    def step(*args, **kwargs):
        assert events[-1] == "clip"
        events.append("step")
        return real_step(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(first.engine, "teacher_force_batch", batch)
        patch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
        patch.setattr(first.optimizer, "step", step)
        report = first.update(windows)
    other = second.update(windows)
    assert events.count("clip") == events.count("step") == 1
    assert events[-2:] == ["clip", "step"]
    assert report["target_rows"] == 19 and report["prefill_rows"] == 139
    assert report["denominator"] == 3 * 128
    assert [window["chunks"] for window in report["windows"]] == [3, 2, 1]
    assert report["sequence_nll"] == pytest.approx(expected_nll, rel=2e-6)
    assert report["loss"] == pytest.approx((expected_nll + coefficient * float(expected_marginal.sum()) / 3) / (3 * 128), rel=2e-6)
    assert report["sequence_nll"] == pytest.approx(other["sequence_nll"], rel=2e-6)
    for name, expected in zip(MARGINAL_NAMES, expected_marginal.tolist()):
        assert report["marginal_nll"][name] == pytest.approx(expected, rel=2e-6)
        assert (report["weighted_marginal_logit_gradient_l2"][name] > 0) == bool(coefficient)
    for a, b in zip(first.engine.model.parameters(), second.engine.model.parameters()):
        if a.grad is None or b.grad is None:
            assert a.grad is b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=3e-7, rtol=2e-5)
    assert any(not torch.equal(a, b) for a, b in zip(model.parameters(), first.engine.model.parameters()))


def test_runner_cuts_writers_at_chunk_boundary_but_trains_history_reads_and_replays_new_versions(monkeypatch):
    window = ragged_windows()[0]
    trainer = SequenceTrainer(small_model(), TrainingConfig(effective_batch_size=1, microbatch_size=1, chunk_rows=4))
    model = trainer.engine.model
    content = model.content_input
    prefill = trainer.engine.prefill
    written, prefix_signatures = [], []

    def observed_content(*args):
        result = content(*args)
        value = result[0]
        if value.requires_grad:
            value.retain_grad()
        written.append(value)
        return result

    def observed_prefill(*args, **kwargs):
        result = prefill(*args, **kwargs)
        prefix_signatures.append(result.signature)
        return result

    monkeypatch.setattr(model, "content_input", observed_content)
    monkeypatch.setattr(trainer.engine, "prefill", observed_prefill)
    trainer.update((window,))
    assert all(not value.requires_grad and value.grad is None for value in written[:window.start])
    target_content = written[window.start:]
    for first, last in ((0, 3), (4, 7), (8, 9)):
        assert target_content[first].grad is not None and target_content[first].grad.norm() > 0
        assert target_content[last].grad is None or not target_content[last].grad.any()
    for layer in model.temporal.layers:
        assert all(parameter.grad is not None and parameter.grad.norm() > 0
                   for parameter in (layer.norm.weight, layer.key.weight, layer.value.weight))
    written.clear()
    trainer.update((window,))
    assert len(prefix_signatures) == 2 and prefix_signatures[0] != prefix_signatures[1]


def test_532_row_target_keeps_the_short_tail_and_clips_once(monkeypatch):
    source = tap_source(562, gap=30)
    sampler = WindowSampler((source,))
    window = sampler.window(source.identity.source_sha256, 30, 2)
    assert window.target_rows == 532
    trainer = SequenceTrainer(small_model(), TrainingConfig(effective_batch_size=1, microbatch_size=1,
                                                          max_grad_norm=.001), ObjectiveConfig(lambda_struct=.25))
    widths, real_batch = [], trainer.engine.teacher_force_batch

    def batch(states, rows):
        widths.append(len(rows[0]))
        return real_batch(states, rows)

    monkeypatch.setattr(trainer.engine, "teacher_force_batch", batch)
    report = trainer.update((window,))
    assert widths == [128, 128, 128, 128, 20]
    assert report["target_rows"] == 532 and report["denominator"] == 128
    assert report["windows"][0]["final_occupancy"] == (False,) * 4
    assert report["clipped"] and report["clipping_frequency"] == 1
    norm = torch.stack([parameter.grad.norm() for parameter in trainer.engine.model.parameters()
                        if parameter.grad is not None]).norm()
    assert norm <= .001001


@pytest.mark.parametrize("chunk_rows", (1, 7, 128))
def test_sequence_cost_is_additive_across_computational_chunks(chunk_rows):
    model = small_model()
    windows = ragged_windows()
    expected, _ = reference_costs(model, windows)
    trainer = SequenceTrainer(model, TrainingConfig(effective_batch_size=3, microbatch_size=2, chunk_rows=chunk_rows))
    assert trainer.update(windows)["sequence_nll"] == pytest.approx(expected, rel=2e-6)


def test_partial_failure_discards_gradients_without_step(monkeypatch):
    trainer = SequenceTrainer(small_model(), TrainingConfig(effective_batch_size=1, microbatch_size=1, chunk_rows=4))
    original = [parameter.detach().clone() for parameter in trainer.engine.model.parameters()]
    real_batch, calls = trainer.engine.teacher_force_batch, 0

    def fail_second(states, chunks):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("chunk failure")
        return real_batch(states, chunks)

    def no_step():
        raise AssertionError("failed accumulation must not step")

    monkeypatch.setattr(trainer.engine, "teacher_force_batch", fail_second)
    monkeypatch.setattr(trainer.optimizer, "step", no_step)
    with pytest.raises(RuntimeError, match="chunk failure"):
        trainer.update((ragged_windows()[0],))
    assert trainer.updates == 0 and not trainer.optimizer.state
    for before, parameter in zip(original, trainer.engine.model.parameters()):
        assert parameter.grad is None
        torch.testing.assert_close(before, parameter, rtol=0, atol=0)


def test_long_note_obligation_survives_every_training_chunk_until_the_true_close(monkeypatch):
    source = admit([(0, 0, 7500)] + [(1 + index % 3, index * 100, index * 100) for index in range(75)])
    sampler = WindowSampler((source,))
    window = sampler.window(source.identity.source_sha256, source.minimum_seed().seed_row_count, 2)
    trainer = SequenceTrainer(small_model(), TrainingConfig(effective_batch_size=1, microbatch_size=1, chunk_rows=7),
                              ObjectiveConfig(lambda_struct=.4))
    real_batch, boundaries = trainer.engine.teacher_force_batch, []

    def batch(states, chunks):
        state = states[0]
        assert state.execution.replay.open_ln_start_ms[0] == 0
        assert state.relation.active_heads[0] == 1
        assert all(token.position.first_id > 1 for token in state.temporal.tokens)
        boundaries.append(state.execution.next_index)
        return real_batch(states, chunks)

    monkeypatch.setattr(trainer.engine, "teacher_force_batch", batch)
    report = trainer.update((window,))
    assert len(boundaries) > 4
    assert report["target_rows"] == window.target_rows
    assert report["windows"][0]["includes_terminal"]
    assert report["windows"][0]["final_occupancy"] == (False,) * 4


def test_effective_batch_split_and_chunk_configuration_are_validated():
    trainer = SequenceTrainer(small_model())
    with pytest.raises(ContractError, match="exactly"):
        trainer.update(ragged_windows())
    source = tap_source(40, split="validation")
    sampler = WindowSampler((source,), split="validation")
    with pytest.raises(ContractError, match="train split"):
        trainer.update((sampler.draw(), sampler.draw()))
    for values in ({"chunk_rows": 129}, {"microbatch_size": 3}, {"effective_batch_size": True},
                   {"learning_rate": 0}, {"max_grad_norm": float("nan")}):
        with pytest.raises(ContractError):
            TrainingConfig(**values)
