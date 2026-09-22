from dataclasses import replace
import json

import pytest

from ensomi_model.research.oracle_time_continuation import verification
from ensomi_model.research.oracle_time_continuation.engine import ContinuationState
from ensomi_model.research.scoped_style_modeling.dataset import ContractError, digest
from .conftest import source_bytes


def test_interval_oracle_checks_full_source_and_short_chart_eligibility():
    objects = [(0, 0, 1000), (1, 0, 1), (2, 0, 100)]
    objects += [(1, index * 10, index * 10) for index in range(1, 40)]
    data = source_bytes(objects)
    result = verification.verify_source(data, digest(data), group_id="song:verified", split="validation")
    assert result["status"] == "passed"
    assert result["source_objects"] == 42
    assert result["source_rows"] == result["verified_pre_post_pairs"] == 42
    assert result["long_notes"] == 3
    assert result["seed"]["seed_note_count"] == 30
    assert result["seed"]["seed_row_count"] == 29
    assert result["seed"]["ineligible_reason"] is None
    assert result["terminal_occupancy"] == [False] * 4
    assert json.loads(json.dumps(result)) == result
    short = source_bytes([(0, 0, 1)])
    report = verification.verify_source(short, digest(short), group_id="song:short", split="train")
    assert report["status"] == "passed"
    assert report["seed"]["ineligible_reason"] == "fewer-than-30-notes"


def test_interval_oracle_detects_incorrect_action_clock(monkeypatch):
    original_commit = ContinuationState.commit

    def incorrect_commit(state, row):
        result = original_commit(state, row)
        if row.time_ms == 10:
            return replace(result, replay=replace(result.replay, last_lane_attack_ms=(0, None, None, None)))
        return result

    monkeypatch.setattr(ContinuationState, "commit", incorrect_commit)
    data = source_bytes([(0, 0, 0), (0, 10, 10)])
    with pytest.raises(ContractError, match="last_lane_attack_ms differs"):
        verification.verify_source(data, digest(data), group_id="song:bad-clock", split="train")
