import numpy as np
import pytest

from ensomi_model.research.audio_skeleton.hydra import compose_config
from ensomi_model.research.audio_skeleton.sensitivity import augment, diagnostics
from ensomi_model.research.bounded_typed_continuation.condition import GenerationCondition
from ensomi_model.research.bounded_typed_continuation.contract import Arm, Timing
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow


def test_added_candidates_preserve_seed_head_times_and_crossing_endpoints():
    condition = GenerationCondition(Arm.R1, Timing((0., 100., 200., 400.), (True, True, False, True)),
                                    (CompleteRow(0., (2, 0, 0, 0)),), (2, None, None, None))
    changed = augment(condition, 1)
    assert changed.seed_rows == condition.seed_rows
    assert changed.timing.times_ms == (0., 50., 100., 150., 200., 300., 400.)
    assert [t for t, h in zip(changed.timing.times_ms, changed.timing.onsets) if h] == [0., 100., 400.]
    assert changed.timing.times_ms[changed.crossing_ends[0]] == 200.
    assert changed.timing.times_ms[-1] == condition.timing.times_ms[-1]
    assert augment(condition, 0) is condition


def test_diagnostics_exclude_seed_born_holds_but_retain_their_action_clocks():
    rows = [CompleteRow(0., (2, 0, 0, 0)), CompleteRow(100., (3, 2, 0, 0)),
            CompleteRow(110., (1, 0, 0, 0)), CompleteRow(300., (0, 3, 0, 0))]
    result = diagnostics(rows, 0.)
    assert result['ln_duration_median_ms'] == 200.
    assert result['ln_heads'] == 1 and result['heads'] == 2
    assert result['below20_per1000heads'] == 500.


def test_packaged_schema_rejects_unknown_and_invalid_fields():
    assert compose_config(['updates=3', 'workers=2']).updates == 3
    for overrides in (['+unused=1'], ['workers=0'], ['run_name=../bad'], ['mode=unknown']):
        with pytest.raises(ValueError):
            compose_config(overrides)
