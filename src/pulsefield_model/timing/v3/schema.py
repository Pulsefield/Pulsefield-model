from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment


TIMING_V3_GRID_SCHEMA = "pulsefield_model.timing_v3_grid_v1"
TIMING_V3_GRID_VERSION = 1
TIMING_V3_MIN_BPM = 20.0
TIMING_V3_MAX_BPM = 1000.0
TIMING_V3_SEAM_ABS_TOLERANCE_MS = 1e-6
TIMING_V3_SEAM_ULP_MULTIPLIER = 8


@dataclass(frozen=True)
class ConstantTimingSection:
    start_beat: int
    end_beat: int
    bpm: float

    def __post_init__(self) -> None:
        start_beat = _require_int(self.start_beat, "start_beat")
        end_beat = _require_int(self.end_beat, "end_beat")
        bpm = _require_bpm(self.bpm, "bpm")

        if end_beat <= start_beat:
            raise ValueError("end_beat must be greater than start_beat")

        object.__setattr__(self, "start_beat", start_beat)
        object.__setattr__(self, "end_beat", end_beat)
        object.__setattr__(self, "bpm", bpm)

    @property
    def beat_count(self) -> int:
        return self.end_beat - self.start_beat

    @property
    def beat_length_ms(self) -> float:
        return 60000.0 / self.bpm

    @property
    def duration_ms(self) -> float:
        return self.beat_count * self.beat_length_ms

    def to_dict(self) -> dict[str, float | int]:
        return {
            "start_beat": self.start_beat,
            "end_beat": self.end_beat,
            "bpm": self.bpm,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConstantTimingSection:
        if not isinstance(payload, Mapping):
            raise ValueError("section payload must be a mapping")
        return cls(
            start_beat=payload["start_beat"],
            end_beat=payload["end_beat"],
            bpm=payload["bpm"],
        )


@dataclass(frozen=True)
class TimingV3Grid:
    origin_beat: int
    origin_time_ms: float
    sections: Sequence[ConstantTimingSection]
    coverage_start_ms: float
    coverage_end_ms: float
    _boundary_times_ms: tuple[float, ...] = field(init=False, repr=False, compare=False)
    _section_start_beats: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        origin_beat = _require_int(self.origin_beat, "origin_beat")
        origin_time_ms = _require_finite_float(self.origin_time_ms, "origin_time_ms")
        coverage_start_ms = _require_finite_float(self.coverage_start_ms, "coverage_start_ms")
        coverage_end_ms = _require_finite_float(self.coverage_end_ms, "coverage_end_ms")
        sections = tuple(_require_section(section, index) for index, section in enumerate(self.sections))

        if not sections:
            raise ValueError("sections must be non-empty")
        if not sections[0].start_beat <= origin_beat < sections[0].end_beat:
            raise ValueError("origin_beat must be inside the first half-open section")
        if coverage_end_ms <= coverage_start_ms:
            raise ValueError("coverage_end_ms must be greater than coverage_start_ms")

        first_boundary_time_ms = (
            origin_time_ms
            + (sections[0].start_beat - origin_beat) * sections[0].beat_length_ms
        )
        if not math.isfinite(first_boundary_time_ms):
            raise ValueError("derived first section boundary time must be finite")

        boundary_times_ms = [float(first_boundary_time_ms)]
        previous_section = sections[0]
        current_time_ms = float(first_boundary_time_ms)

        for index, section in enumerate(sections):
            if index > 0 and previous_section.end_beat != section.start_beat:
                raise ValueError("sections must be contiguous on the integer beat axis")

            next_time_ms = current_time_ms + section.duration_ms
            if not math.isfinite(next_time_ms) or next_time_ms <= current_time_ms:
                raise ValueError("derived section boundary times must be finite and strictly increasing")

            boundary_times_ms.append(float(next_time_ms))
            current_time_ms = float(next_time_ms)
            previous_section = section

        if coverage_start_ms < boundary_times_ms[0] - TIMING_V3_SEAM_ABS_TOLERANCE_MS:
            raise ValueError("coverage_start_ms is before the first derived section boundary")
        if coverage_end_ms > boundary_times_ms[-1] + TIMING_V3_SEAM_ABS_TOLERANCE_MS:
            raise ValueError("coverage_end_ms is after the final derived section boundary")

        object.__setattr__(self, "origin_beat", origin_beat)
        object.__setattr__(self, "origin_time_ms", origin_time_ms)
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "coverage_start_ms", coverage_start_ms)
        object.__setattr__(self, "coverage_end_ms", coverage_end_ms)
        object.__setattr__(self, "_boundary_times_ms", tuple(boundary_times_ms))
        object.__setattr__(self, "_section_start_beats", tuple(section.start_beat for section in sections))

    @property
    def start_beat(self) -> int:
        return self.sections[0].start_beat

    @property
    def end_beat(self) -> int:
        return self.sections[-1].end_beat

    @property
    def start_time_ms(self) -> float:
        return self._boundary_times_ms[0]

    @property
    def end_time_ms(self) -> float:
        return self._boundary_times_ms[-1]

    @property
    def grid_end_time_ms(self) -> float:
        return self.end_time_ms

    @property
    def boundary_times_ms(self) -> tuple[float, ...]:
        return self._boundary_times_ms

    @property
    def section_start_times_ms(self) -> tuple[float, ...]:
        return self._boundary_times_ms[:-1]

    @property
    def section_end_times_ms(self) -> tuple[float, ...]:
        return self._boundary_times_ms[1:]

    def section_start_time_ms(self, section: int | ConstantTimingSection) -> float:
        return self._boundary_times_ms[self._section_index_for_argument(section)]

    def section_end_time_ms(self, section: int | ConstantTimingSection) -> float:
        return self._boundary_times_ms[self._section_index_for_argument(section) + 1]

    def section_index_at_beat(self, beat: float) -> int:
        beat_value = _require_finite_real(beat, "beat")
        if beat_value < self.start_beat or beat_value >= self.end_beat:
            raise ValueError("beat is outside the half-open grid beat range")
        return bisect.bisect_right(self._section_start_beats, beat_value) - 1

    def section_at_beat(self, beat: float) -> ConstantTimingSection:
        return self.sections[self.section_index_at_beat(beat)]

    def lookup_section_by_beat(self, beat: float) -> ConstantTimingSection:
        return self.section_at_beat(beat)

    def section_for_beat(self, beat: float) -> ConstantTimingSection:
        return self.section_at_beat(beat)

    def section_index_at_time(self, time_ms: float) -> int:
        time_value_ms = _require_finite_float(time_ms, "time_ms")
        if time_value_ms < self.start_time_ms or time_value_ms >= self.end_time_ms:
            raise ValueError("time_ms is outside the half-open grid time range")
        return bisect.bisect_right(self._boundary_times_ms, time_value_ms) - 1

    def section_at_time(self, time_ms: float) -> ConstantTimingSection:
        return self.sections[self.section_index_at_time(time_ms)]

    def lookup_section_by_time(self, time_ms: float) -> ConstantTimingSection:
        return self.section_at_time(time_ms)

    def section_for_time(self, time_ms: float) -> ConstantTimingSection:
        return self.section_at_time(time_ms)

    def time_at_beat(self, beat: float) -> float:
        beat_value = _require_finite_real(beat, "beat")
        if beat_value < self.start_beat or beat_value > self.end_beat:
            raise ValueError("beat is outside the closed grid beat boundary range")
        if beat_value == self.end_beat:
            return self.end_time_ms

        section_index = self.section_index_at_beat(beat_value)
        section = self.sections[section_index]
        return self._boundary_times_ms[section_index] + (beat_value - section.start_beat) * section.beat_length_ms

    def beat_at_time(self, time_ms: float) -> float:
        time_value_ms = _require_finite_float(time_ms, "time_ms")
        if time_value_ms < self.start_time_ms or time_value_ms > self.end_time_ms:
            raise ValueError("time_ms is outside the closed grid time boundary range")
        if time_value_ms == self.end_time_ms:
            return float(self.end_beat)

        section_index = self.section_index_at_time(time_value_ms)
        section = self.sections[section_index]
        return section.start_beat + (time_value_ms - self._boundary_times_ms[section_index]) / section.beat_length_ms

    def to_fitted_timing_grid(self) -> FittedTimingGrid:
        return FittedTimingGrid(
            tuple(
                TimingSegment(offset_ms=self._boundary_times_ms[index], beat_length_ms=section.beat_length_ms)
                for index, section in enumerate(self.sections)
            )
        )

    def to_human_lines(self) -> tuple[str, ...]:
        return tuple(
            f"beat [{section.start_beat},{section.end_beat}) bpm {_format_float(section.bpm)}"
            for section in self.sections
        )

    def to_human_text(self) -> str:
        return "\n".join(self.to_human_lines())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TIMING_V3_GRID_SCHEMA,
            "version": TIMING_V3_GRID_VERSION,
            "origin_beat": self.origin_beat,
            "origin_time_ms": self.origin_time_ms,
            "coverage_start_ms": self.coverage_start_ms,
            "coverage_end_ms": self.coverage_end_ms,
            "sections": [section.to_dict() for section in self.sections],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TimingV3Grid:
        if not isinstance(payload, Mapping):
            raise ValueError("grid payload must be a mapping")
        if payload.get("schema") != TIMING_V3_GRID_SCHEMA:
            raise ValueError(f"schema must be {TIMING_V3_GRID_SCHEMA!r}")
        version = _require_int(payload.get("version"), "version")
        if version != TIMING_V3_GRID_VERSION:
            raise ValueError(f"version must be {TIMING_V3_GRID_VERSION!r}")

        sections_payload = payload["sections"]
        if not isinstance(sections_payload, Sequence) or isinstance(sections_payload, (str, bytes)):
            raise ValueError("sections must be a sequence")

        return cls(
            origin_beat=payload["origin_beat"],
            origin_time_ms=payload["origin_time_ms"],
            sections=tuple(ConstantTimingSection.from_dict(section) for section in sections_payload),
            coverage_start_ms=payload["coverage_start_ms"],
            coverage_end_ms=payload["coverage_end_ms"],
        )

    def _section_index_for_argument(self, section: int | ConstantTimingSection) -> int:
        if isinstance(section, Integral) and not isinstance(section, bool):
            index = int(section)
            if index < 0:
                index += len(self.sections)
            if index < 0 or index >= len(self.sections):
                raise IndexError("section index out of range")
            return index

        if isinstance(section, ConstantTimingSection):
            for index, candidate in enumerate(self.sections):
                if candidate == section:
                    return index
            raise ValueError("section does not belong to this grid")

        raise TypeError("section must be an integer index or ConstantTimingSection")


def roundtrip_seam_tolerance_ms(time_ms: float) -> float:
    value = _require_finite_float(time_ms, "time_ms")
    return max(TIMING_V3_SEAM_ABS_TOLERANCE_MS, TIMING_V3_SEAM_ULP_MULTIPLIER * math.ulp(value))


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _require_finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_finite_float(value: object, name: str) -> float:
    return float(_require_finite_real(value, name))


def _require_bpm(value: object, name: str) -> float:
    bpm = _require_finite_float(value, name)
    if bpm < TIMING_V3_MIN_BPM or bpm > TIMING_V3_MAX_BPM:
        raise ValueError(f"{name} must be within [{TIMING_V3_MIN_BPM:g}, {TIMING_V3_MAX_BPM:g}]")
    return bpm


def _require_section(section: object, index: int) -> ConstantTimingSection:
    if not isinstance(section, ConstantTimingSection):
        raise ValueError(f"sections[{index}] must be a ConstantTimingSection")
    return section


def _format_float(value: float) -> str:
    return f"{value:.15g}"
