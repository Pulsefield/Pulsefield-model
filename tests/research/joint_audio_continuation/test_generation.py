from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.joint_audio_continuation.generation import (
    NativeGeneration, load_model, rollout, save_rollout,
)
from ensomi_model.research.joint_audio_continuation.model import JointAudioModel, JointModelConfig
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.scoped_style_modeling.dataset import ContractError


def config():
    return JointModelConfig(hidden=12, audio_width=8, audio_levels=2, history_levels=2,
                            expansion=2, coupling_rank=3, routing_hidden=20, release_hidden=24)


class HoldThenSilence(JointAudioModel):
    def timing_logits(self, audio, history, exact):
        logits = audio.new_full((*audio.shape[:2], 10), -torch.inf)
        if bool(exact[0, 0, 0, -1]):
            logits[0, 0, 0] = torch.inf
        return logits

    def row_log_probs(self, audio, history, exact, legal, occupancy):
        logits = audio.new_full((1, 256), -torch.inf)
        action = (3, 0, 0, 0) if bool(occupancy[0, 0]) else (2, 0, 0, 0)
        index = ROW_ACTIONS.index(action)
        assert bool(legal[0, index])
        logits[0, index] = 0.
        return logits


class EveryMillisecond(JointAudioModel):
    def timing_logits(self, audio, history, exact):
        return audio.new_full((*audio.shape[:2], 10), torch.inf)

    def row_log_probs(self, audio, history, exact, legal, occupancy):
        logits = audio.new_full((1, 256), -torch.inf)
        logits[0, ROW_ACTIONS.index((1, 0, 0, 0))] = 0.
        return logits


@pytest.mark.parametrize('chunk_ms', [7, 80, 1000])
def test_bos_event_at_zero_and_hold_survives_chunks_until_true_terminal(chunk_ms):
    model = HoldThenSilence(config())
    result = rollout(model, np.zeros((13, 128), np.float32), 123, chunk_ms=chunk_ms)
    assert result.completed and result.stop_reason == 'completed' and result.coverage_ms == 123
    assert result.rows == (CompleteRow(0, (2, 0, 0, 0)), CompleteRow(123, (3, 0, 0, 0)))
    assert result.metrics['forced_terminal_events'] == 1
    assert result.metrics['startup_target_ms'] == 123
    assert result.metrics['startup_seconds'] >= result.metrics['audio_encode_seconds']
    assert result.metrics['open_lanes'] == [False] * 4


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='requires Apple MPS'))])
def test_native_random_rollout_preserves_draw_across_scheduler_partitions(device):
    torch.manual_seed(67)
    model = JointAudioModel(config()).to(device)
    mel = np.random.default_rng(61).normal(size=(201, 128)).astype(np.float32)
    a = rollout(model, mel, 2000, seed=83, chunk_ms=37)
    b = rollout(model, mel, 2000, seed=83, chunk_ms=4000)
    assert a.completed and b.completed and a.rows == b.rows
    assert all(int(row.time_ms) == row.time_ms for row in a.rows)
    assert a.metrics['scheduler_steps'] != b.metrics['scheduler_steps']


def test_row_or_time_cap_never_fabricates_an_endpoint(tmp_path):
    model = HoldThenSilence(config())
    mel = np.zeros((25, 128), np.float32)
    result = rollout(model, mel, 249, chunk_ms=50, max_rows=1)
    assert not result.completed and result.stop_reason == 'row_limit'
    assert result.rows == (CompleteRow(0, (2, 0, 0, 0)),)
    info = save_rollout(tmp_path / 'capped', result)
    assert info['mechanics']['open_heads'] == [0., None, None, None]
    assert not (tmp_path / 'capped' / 'generated.osu').exists()
    timed = rollout(model, mel, 249, max_seconds=1e-12)
    assert not timed.completed and timed.stop_reason == 'time_limit' and not timed.rows
    assert timed.metrics['startup_seconds'] is None


def test_native_clock_has_no_ten_ms_spacing_or_peak_suppression():
    result = rollout(EveryMillisecond(config()), np.zeros((3, 128), np.float32), 4, chunk_ms=2)
    assert result.completed
    assert [row.time_ms for row in result.rows] == [0., 1., 2., 3., 4.]


def test_resource_callback_checks_every_twenty_steps_without_changing_draws():
    calls = []

    def stop():
        calls.append(True)
        return 'pause_file' if len(calls) == 2 else None

    result = rollout(EveryMillisecond(config()), np.zeros((6, 128), np.float32), 49,
                     chunk_ms=8, stop_callback=stop)
    assert not result.completed and result.stop_reason == 'pause_file'
    assert len(calls) == 2 and len(result.rows) == 20
    assert [row.time_ms for row in result.rows] == list(range(20))


def test_incremental_updates_publish_unresolved_heads_and_empty_coverage_before_terminal():
    updates = []
    result = rollout(HoldThenSilence(config()), np.zeros((13, 128), np.float32), 123,
                     chunk_ms=25, on_update=updates.append)
    assert updates[0].row == CompleteRow(0, (2, 0, 0, 0))
    assert not updates[0].completed
    assert any(u.row is None and u.coverage_ms < 123 for u in updates)
    assert [u.coverage_ms for u in updates] == sorted({u.coverage_ms for u in updates})
    assert tuple(u.row for u in updates if u.row is not None) == result.rows
    assert updates[-1].row == CompleteRow(123, (3, 0, 0, 0)) and updates[-1].completed
    capped = []
    result = rollout(HoldThenSilence(config()), np.zeros((13, 128), np.float32), 123,
                     max_rows=1, on_update=capped.append)
    assert not result.completed and len(capped) == 1 and not capped[0].completed


def test_publication_callback_does_not_change_draws_and_consumer_errors_propagate():
    torch.manual_seed(37)
    model = JointAudioModel(config())
    mel = np.zeros((51, 128), np.float32)
    expected = rollout(model, mel, 500, seed=71, chunk_ms=71)
    updates = []
    actual = rollout(model, mel, 500, seed=71, chunk_ms=71, on_update=updates.append)
    assert actual.rows == expected.rows
    assert tuple(u.row for u in updates if u.row is not None) == actual.rows
    def fail(update):
        raise RuntimeError('consumer disconnected')
    with pytest.raises(RuntimeError, match='consumer disconnected'):
        rollout(model, mel, 500, on_update=fail)


def test_no_event_song_has_fixed_empty_coverage_without_a_terminal_tap():
    model = JointAudioModel(config())
    with torch.no_grad():
        model.timing[-1].weight.zero_()
        model.timing[-1].bias.fill_(-1000.)
    result = rollout(model, np.zeros((801, 128), np.float32), 8001, chunk_ms=2000)
    assert result.completed and not result.rows and result.coverage_ms == 8001
    assert result.metrics['startup_target_ms'] == 8000
    assert result.metrics['startup_seconds'] is not None
    assert result.metrics['forced_terminal_events'] == 0


def test_complete_export_round_trips_and_bundles_paired_audio(tmp_path):
    rows = (CompleteRow(0, (2, 1, 0, 0)), CompleteRow(13, (0, 0, 1, 0)),
            CompleteRow(123, (3, 0, 0, 0)))
    source = tmp_path / 'source.osu'
    source.write_text('osu file format v14\n[General]\nAudioFilename:original.mp3\nMode:3\n'
                      '[Metadata]\nTitle:Test\nArtist:Test\nVersion:Source\n'
                      '[Difficulty]\nCircleSize:4\n[TimingPoints]\n0,500,4,2,0,100,1,0\n'
                      '[HitObjects]\n64,192,9000,1,0,0:0:0:0:\n')
    audio = tmp_path / 'original.mp3'
    audio.write_bytes(b'paired-audio-fixture')
    result = NativeGeneration(rows, True, 'completed', 123, {})
    info = save_rollout(tmp_path / 'complete', result, source_file=source, audio_file=audio)
    assert info['reparse_pass']
    assert info['mechanics']['rows'] == 3 and info['mechanics']['ln_closes'] == 1
    assert (tmp_path / 'complete' / 'audio.mp3').read_bytes() == audio.read_bytes()
    text = (tmp_path / 'complete' / 'generated.osu').read_text()
    assert 'AudioFilename:audio.mp3' in text and '9000' not in text
    assert 'joint audio native' in text and 'oracle continuation' not in text
    persisted = [json.loads(line) for line in (tmp_path / 'complete' / 'rows.jsonl').read_text().splitlines()]
    assert [row['event_id'] for row in persisted] == [0, 1, 2]
    with pytest.raises(FileExistsError):
        save_rollout(tmp_path / 'complete', result)


def test_loader_pins_checkpoint_and_preserves_normalization_buffers(tmp_path):
    model = JointAudioModel(config())
    model.set_audio_normalization(torch.arange(128, dtype=torch.float32), torch.full((128,), 2.))
    payload = dict(model_config=asdict(config()), model=model.state_dict(),
                   source_revision='a' * 40, manifest_sha256='b' * 64, config=dict(seed=17),
                   rng_state=torch.get_rng_state())
    path = tmp_path / 'model.pt'
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    restored, metadata = load_model(path, digest)
    torch.testing.assert_close(restored.audio_mean, model.audio_mean)
    torch.testing.assert_close(restored.audio_std, model.audio_std)
    assert metadata['manifest_sha256'] == 'b' * 64
    json.dumps(metadata)
    with pytest.raises(ContractError, match='SHA-256'):
        load_model(path, '0' * 64)
