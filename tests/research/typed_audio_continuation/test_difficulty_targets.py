import numpy as np
import pytest

from ensomi_model.osu_core.difficulty import RawHitObject, ManiaStrain, create_difficulty_hit_objects
from ensomi_model.research.oracle_time_continuation.storage import ROW_DTYPE
from ensomi_model.research.typed_audio_continuation.controls import source_schedule
from ensomi_model.research.typed_audio_continuation.difficulty_targets import ScopeStrainTrace


def test_trace_reuses_full_prefix_with_real_ln_endpoints():
    objects = [RawHitObject(0, 0, 0), RawHitObject(100, 900, 1),
               RawHitObject(200, 200, 0), RawHitObject(300, 300, 2),
               RawHitObject(500, 1000, 3), RawHitObject(600, 600, 0)]
    trace = ScopeStrainTrace.from_objects(objects)
    skill = ManiaStrain(4)
    for obj in create_difficulty_hit_objects(objects, 4, 1.):
        skill.process(obj)
    np.testing.assert_allclose(trace.at(trace.times), skill.object_strains)
    # A slice must inherit the state produced by the earlier held and tapped
    # objects. Rebuilding the slice silently loses those relationships.
    cropped = ScopeStrainTrace.from_objects([o for o in objects if o.start_time >= 500])
    assert trace.level(500, 800) > cropped.level(500, 800)
    future = ScopeStrainTrace.from_objects(objects+[RawHitObject(800, 800, 1)])
    assert future.level(500, 800) == pytest.approx(trace.level(500, 800))


def test_peak_windows_keep_incoming_decay_without_copying_old_peaks():
    trace = ScopeStrainTrace(np.array([100., 700.]), np.array([8., 20.]), np.array([12., 30.]))
    assert trace.peaks(300, 700).tolist() == pytest.approx(trace.at([300]).tolist())
    assert trace.peaks(300, 701)[-1] == 50
    assert trace.level(300, 700) < trace.level(100, 300)
    empty = ScopeStrainTrace.from_objects([])
    assert empty.level(0, 32000) == 0


def test_source_scope_labels_follow_local_demand_with_shared_condition_layout():
    objects = [RawHitObject(t, t, i % 4) for i, t in enumerate(range(100, 2000, 80))]
    trace = ScopeStrainTrace.from_objects(objects)
    rows = np.zeros(2, dtype=ROW_DTYPE)
    rows['time'] = [100, 1100]
    rows['actions'][:, 0] = 1
    baseline = source_schedule(rows, 4000, 4., 1000)
    scoped = source_schedule(rows, 4000, 4., 1000, difficulty_trace=trace)
    values = scoped.resolved_ranges(0, 4001)
    assert values[1].stars > values[3].stars
    assert values[1].stars == pytest.approx(trace.level(1000, 2000))
    assert all(s.stars == 4. for s in baseline.resolved_ranges(0, 4001))
    assert scoped.width == baseline.width
    assert scoped.at([1100])[0, 2] == baseline.at([1100])[0, 2] == 1
