import json

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.features import content_features
from ensomi_model.research.joint_audio_continuation.generation import (
    ObservedPrefix, _prefix_context, rollout, save_rollout,
)
from ensomi_model.research.joint_audio_continuation.model import JointAudioModel
from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from .test_generation import config, EveryMillisecond, HoldThenSilence


def test_finite_prefill_preserves_old_hold_clocks_and_matches_full_cached_history():
    torch.manual_seed(91)
    model = JointAudioModel(config()).eval()
    rows = (CompleteRow(0, (2, 0, 0, 0)),) + tuple(
        CompleteRow(i*19+i%3, (0, 1, 0, 0)) for i in range(1, 41))
    prefix = ObservedPrefix(rows, 800)
    replay, cache = _prefix_context(model, prefix)
    expected, full = ExactReplayState(), model.temporal.empty_cache()
    raw = []
    for row in rows:
        previous = expected.last_row.time_ms if expected.last_row else None
        expected = commit(expected, row)
        raw.append(content_features([row], [previous], [[None]*4])[0])
        full = model.temporal.append(full, torch.from_numpy(raw[-1]))
    assert replay == expected and replay.open_ln_start_ms == (0., None, None, None)
    assert replay.row_count == 41 and replay.clocks_at(800).ln_age_ms[0] == 800
    limit = model.temporal.config.receptive_tokens
    assert cache.rows == limit < len(rows) and cache.truncated_start
    torch.testing.assert_close(model.temporal.read(cache), model.temporal.read(full), atol=2e-5, rtol=2e-5)
    with torch.no_grad():
        dense = model.encode_history(torch.from_numpy(np.asarray(raw[-limit:]))[None],
            torch.ones(1, limit, dtype=torch.bool), torch.tensor([True]))
    torch.testing.assert_close(model.temporal.read(cache), dense[0], atol=2e-5, rtol=2e-5)
    # The rebuilt buffers must also be correct for later steps, not just read().
    for row in (CompleteRow(811, (0,0,1,0)), CompleteRow(879, (3,0,0,0))):
        value = torch.from_numpy(content_features([row], [expected.last_row.time_ms], [[None]*4])[0])
        cache, full = model.temporal.append(cache,value), model.temporal.append(full,value)
        expected = commit(expected,row)
        torch.testing.assert_close(model.temporal.read(cache), model.temporal.read(full), atol=2e-5, rtol=2e-5)


def test_prefix_known_silence_and_open_hold_are_not_fabricated_rows_or_endpoints():
    prefix = ObservedPrefix((CompleteRow(0, (2,0,0,0)),), 50)
    updates = []
    result = rollout(HoldThenSilence(config()), np.zeros((13,128),np.float32),123,
                     prefix=prefix,chunk_ms=25,on_update=updates.append)
    assert result.rows == (prefix.rows[0], CompleteRow(123,(3,0,0,0)))
    assert result.completed and result.metrics['generated_heads'] == 0
    assert result.metrics['prefix_heads'] == 1 and result.metrics['generated_rows'] == 1
    assert result.metrics['startup_target_ms'] == 123
    assert result.metrics['first30_heads_seconds'] is None
    assert updates[0].row is None and updates[0].coverage_ms == 75
    assert tuple(u.row for u in updates if u.row is not None) == result.rows[1:]


def test_caps_and_first30_count_new_material_only_and_export_marks_observed_prefix(tmp_path):
    prefix = ObservedPrefix(tuple(CompleteRow(i,(1,0,0,0)) for i in range(40)),39)
    model,mel = EveryMillisecond(config()),np.zeros((10,128),np.float32)
    capped = rollout(model,mel,79,prefix=prefix,max_rows=3)
    assert not capped.completed and capped.stop_reason == 'row_limit'
    assert capped.rows[40:] == tuple(CompleteRow(i,(1,0,0,0)) for i in (40,41,42))
    assert capped.metrics['prefix_heads'] == 40 and capped.metrics['generated_heads'] == 3
    assert capped.metrics['first30_heads_through_ms'] is None
    complete = rollout(model,mel,79,prefix=prefix)
    assert complete.metrics['first30_heads_through_ms'] == 69
    assert complete.metrics['startup_target_ms'] == 79
    info = save_rollout(tmp_path/'prefix',complete)
    rows = [json.loads(line) for line in (tmp_path/'prefix/rows.jsonl').read_text().splitlines()]
    assert all(r['origin']=='observed' for r in rows[:40])
    assert all(r['origin']=='sampled' for r in rows[40:])
    assert 'Observed prefix through 39 ms + model continuation' in (tmp_path/'prefix/generated.osu').read_text()
    assert info['reparse_pass'] and info['prefix_rows'] == 40


def test_empty_observed_coverage_starts_at_next_millisecond_without_bos_replay():
    result = rollout(EveryMillisecond(config()),np.zeros((3,128),np.float32),14,
                     prefix=ObservedPrefix((),10))
    assert [r.time_ms for r in result.rows] == [11.,12.,13.,14.]
    assert result.metrics['prefix_rows'] == 0 and result.metrics['prefix_coverage_ms'] == 10


def test_explicit_empty_bos_preserves_sampling_and_row_bytes(tmp_path):
    torch.manual_seed(193)
    model = JointAudioModel(config())
    mel = np.random.default_rng(2).normal(size=(51,128)).astype(np.float32)
    expected = rollout(model,mel,500,seed=97,chunk_ms=83)
    actual = rollout(model,mel,500,seed=97,chunk_ms=83,prefix=ObservedPrefix((),-1))
    assert actual.rows == expected.rows and actual.metrics['startup_target_ms'] == 500
    a,b = save_rollout(tmp_path/'a',actual),save_rollout(tmp_path/'b',expected)
    assert a['rows_sha256'] == b['rows_sha256']
    assert 'origin' not in (tmp_path/'a/rows.jsonl').read_text()


@pytest.mark.parametrize('rows,coverage', [([CompleteRow(.5,(1,0,0,0))],1),
    ([CompleteRow(2,(1,0,0,0))],1), ([],True), ([], -2), ([object()],1)])
def test_invalid_prefix_clock_or_row_is_rejected(rows,coverage):
    with pytest.raises(ContractError,match='Observed prefix'):
        ObservedPrefix(rows,coverage)


@pytest.mark.parametrize('rows', [
    (CompleteRow(1,(1,0,0,0)),CompleteRow(1,(0,1,0,0))),
    (CompleteRow(1,(3,0,0,0)),),
    (CompleteRow(1,(2,0,0,0)),CompleteRow(2,(1,0,0,0))),
])
def test_invalid_prefix_replay_is_rejected(rows):
    with pytest.raises(ContractError):
        rollout(JointAudioModel(config()),np.zeros((2,128),np.float32),10,
                prefix=ObservedPrefix(rows,2))


def test_prefix_cannot_claim_real_terminal_or_publish_on_resource_cap():
    model = HoldThenSilence(config())
    mel = np.zeros((13,128),np.float32)
    with pytest.raises(ContractError,match='precede true audio end'):
        rollout(model,mel,123,prefix=ObservedPrefix((),123))
    updates = []
    prefix = ObservedPrefix((CompleteRow(0,(2,0,0,0)),),50)
    result = rollout(model,mel,123,prefix=prefix,max_seconds=1e-12,on_update=updates.append)
    assert result.rows == prefix.rows and not result.completed and not updates
    assert result.metrics['open_lanes'] == [True,False,False,False]
    assert result.metrics['coverage'] == []
