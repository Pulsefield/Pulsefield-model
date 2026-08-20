from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Mapping, Sequence, TypeAlias

from pulsefield_model.timing.v3.schema import TIMING_V3_MAX_BPM, TIMING_V3_MIN_BPM


ANALYTIC_CURVE_VERSION = "pulsefield_model.timing_v3_analytic_curve_v1"
TIMING_V3_ANALYTIC_CURVE_VERSION = ANALYTIC_CURVE_VERSION
_RAMP_ENDPOINT_TOLERANCE_SECONDS = 1e-12


@dataclass(frozen=True)
class ConstantTempoSection:
    """A constant-tempo half-open interval on the global beat axis."""

    start_beat: int
    end_beat: int
    bpm: float
    _duration_seconds: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        start_beat = _require_int(self.start_beat, "start_beat")
        end_beat = _require_int(self.end_beat, "end_beat")
        bpm = _require_bpm(self.bpm, "bpm")
        if end_beat <= start_beat:
            raise ValueError("end_beat must be greater than start_beat")

        duration_seconds = _constant_duration_seconds(end_beat - start_beat, bpm)
        object.__setattr__(self, "start_beat", start_beat)
        object.__setattr__(self, "end_beat", end_beat)
        object.__setattr__(self, "bpm", bpm)
        object.__setattr__(self, "_duration_seconds", duration_seconds)

    @property
    def beat_count(self) -> int:
        return self.end_beat - self.start_beat

    @property
    def duration_seconds(self) -> float:
        return self._duration_seconds

    @property
    def duration_ms(self) -> float:
        return self._duration_seconds * 1000.0

    @property
    def start_bpm(self) -> float:
        return self.bpm

    @property
    def end_bpm(self) -> float:
        return self.bpm

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "constant",
            "start_beat": self.start_beat,
            "end_beat": self.end_beat,
            "bpm": _canonical_float(self.bpm),
        }


@dataclass(frozen=True)
class LinearTimeRampSection:
    """A half-open interval whose BPM changes linearly in wall-clock time."""

    start_beat: int
    end_beat: int
    start_bpm: float
    end_bpm: float
    _duration_seconds: float = field(init=False, repr=False, compare=False)
    _acceleration_bpm_per_second: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        start_beat = _require_int(self.start_beat, "start_beat")
        end_beat = _require_int(self.end_beat, "end_beat")
        start_bpm = _require_bpm(self.start_bpm, "start_bpm")
        end_bpm = _require_bpm(self.end_bpm, "end_bpm")
        if end_beat <= start_beat:
            raise ValueError("end_beat must be greater than start_beat")
        if start_bpm == end_bpm:
            raise ValueError(
                "start_bpm and end_bpm must differ; use ConstantTempoSection "
                "for equal endpoint BPM"
            )

        duration_seconds = _ramp_duration_seconds(
            end_beat - start_beat,
            start_bpm,
            end_bpm,
        )
        acceleration = (end_bpm - start_bpm) / duration_seconds
        if not math.isfinite(acceleration):
            raise ValueError("derived ramp acceleration must be finite")
        if acceleration == 0.0:
            raise ValueError(
                "endpoint BPM difference is too small for a nonzero float64 ramp; "
                "use ConstantTempoSection"
            )

        object.__setattr__(self, "start_beat", start_beat)
        object.__setattr__(self, "end_beat", end_beat)
        object.__setattr__(self, "start_bpm", start_bpm)
        object.__setattr__(self, "end_bpm", end_bpm)
        object.__setattr__(self, "_duration_seconds", duration_seconds)
        object.__setattr__(self, "_acceleration_bpm_per_second", acceleration)

    @property
    def beat_count(self) -> int:
        return self.end_beat - self.start_beat

    @property
    def duration_seconds(self) -> float:
        return self._duration_seconds

    @property
    def duration_ms(self) -> float:
        return self._duration_seconds * 1000.0

    @property
    def acceleration_bpm_per_second(self) -> float:
        return self._acceleration_bpm_per_second

    def elapsed_seconds_at_beat(self, beat: float) -> float:
        beat_value = _require_finite_real(beat, "beat")
        if beat_value < self.start_beat or beat_value > self.end_beat:
            raise ValueError("beat is outside the closed section beat domain")
        elapsed = solve_linear_time_ramp_elapsed_seconds(
            beat_value - self.start_beat,
            start_bpm=self.start_bpm,
            acceleration_bpm_per_second=self.acceleration_bpm_per_second,
        )
        return _clamp_ramp_elapsed_seconds(elapsed, self.duration_seconds)

    def beat_at_elapsed_seconds(self, elapsed_seconds: float) -> float:
        elapsed = _require_finite_float(elapsed_seconds, "elapsed_seconds")
        elapsed = _clamp_ramp_elapsed_seconds(elapsed, self.duration_seconds)
        if elapsed == 0.0:
            return float(self.start_beat)
        if elapsed == self.duration_seconds:
            return float(self.end_beat)
        beat_offset = math.fsum(
            (
                self.start_bpm * elapsed,
                0.5 * self.acceleration_bpm_per_second * elapsed * elapsed,
            )
        ) / 60.0
        beat = self.start_beat + beat_offset
        if not math.isfinite(beat):
            raise ValueError("derived beat must be finite")
        return beat

    def bpm_at_elapsed_seconds(self, elapsed_seconds: float) -> float:
        elapsed = _require_finite_float(elapsed_seconds, "elapsed_seconds")
        elapsed = _clamp_ramp_elapsed_seconds(elapsed, self.duration_seconds)
        if elapsed == 0.0:
            return self.start_bpm
        if elapsed == self.duration_seconds:
            return self.end_bpm
        bpm = self.start_bpm + self.acceleration_bpm_per_second * elapsed
        if not math.isfinite(bpm):
            raise ValueError("derived BPM must be finite")
        return bpm

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "linear_bpm_time",
            "start_beat": self.start_beat,
            "end_beat": self.end_beat,
            "start_bpm": _canonical_float(self.start_bpm),
            "end_bpm": _canonical_float(self.end_bpm),
        }


TimingCurveSection: TypeAlias = ConstantTempoSection | LinearTimeRampSection


@dataclass(frozen=True)
class TimingCurveSeam:
    beat: int
    time_ms: float
    left_bpm: float
    right_bpm: float
    tempo_continuous: bool
    phase_discontinuity_ms: float = 0.0


@dataclass(frozen=True)
class PhaseContinuousTimingCurve:
    """An analytic tempo curve with one anchor and one cumulative beat phase."""

    origin_beat: int
    origin_time_ms: float
    sections: Sequence[TimingCurveSection]
    _boundary_times_ms: tuple[float, ...] = field(init=False, repr=False, compare=False)
    _section_start_beats: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        origin_beat = _require_int(self.origin_beat, "origin_beat")
        origin_time_ms = _require_finite_float(self.origin_time_ms, "origin_time_ms")
        sections = tuple(_require_section(value, index) for index, value in enumerate(self.sections))
        if not sections:
            raise ValueError("sections must be non-empty")
        if origin_beat != sections[0].start_beat:
            raise ValueError("origin_beat must equal the first section's start_beat")

        boundary_times_ms = [origin_time_ms]
        for index, section in enumerate(sections):
            if index:
                previous = sections[index - 1]
                if previous.end_beat != section.start_beat:
                    raise ValueError("sections must be contiguous on the integer beat axis")
                if (
                    isinstance(previous, ConstantTempoSection)
                    and isinstance(section, ConstantTempoSection)
                    and previous.bpm == section.bpm
                ):
                    raise ValueError(
                        "adjacent same-BPM constant sections are noncanonical; merge them"
                    )

            end_time_ms = boundary_times_ms[-1] + section.duration_ms
            if not math.isfinite(end_time_ms) or end_time_ms <= boundary_times_ms[-1]:
                raise ValueError("derived section boundary times must be finite and increasing")
            boundary_times_ms.append(float(end_time_ms))

        object.__setattr__(self, "origin_beat", origin_beat)
        object.__setattr__(self, "origin_time_ms", origin_time_ms)
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "_boundary_times_ms", tuple(boundary_times_ms))
        object.__setattr__(
            self,
            "_section_start_beats",
            tuple(section.start_beat for section in sections),
        )

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
    def boundary_times_ms(self) -> tuple[float, ...]:
        return self._boundary_times_ms

    @property
    def section_start_times_ms(self) -> tuple[float, ...]:
        return self._boundary_times_ms[:-1]

    @property
    def section_end_times_ms(self) -> tuple[float, ...]:
        return self._boundary_times_ms[1:]

    @property
    def curve_class(self) -> str:
        if any(isinstance(section, LinearTimeRampSection) for section in self.sections):
            return "ramp"
        if len(self.sections) >= 2:
            return "jump"
        return "constant"

    @property
    def classification(self) -> str:
        return self.curve_class

    @property
    def seam_reports(self) -> tuple[TimingCurveSeam, ...]:
        reports: list[TimingCurveSeam] = []
        for index in range(1, len(self.sections)):
            left = self.sections[index - 1]
            right = self.sections[index]
            left_bpm = left.end_bpm
            right_bpm = right.start_bpm
            reports.append(
                TimingCurveSeam(
                    beat=right.start_beat,
                    time_ms=self._boundary_times_ms[index],
                    left_bpm=left_bpm,
                    right_bpm=right_bpm,
                    tempo_continuous=left_bpm == right_bpm,
                )
            )
        return tuple(reports)

    def section_index_at_beat(self, beat: float) -> int:
        beat_value = _require_finite_real(beat, "beat")
        if beat_value < self.start_beat or beat_value > self.end_beat:
            raise ValueError("beat is outside the closed curve beat domain")
        if beat_value == self.end_beat:
            return len(self.sections) - 1
        return bisect.bisect_right(self._section_start_beats, beat_value) - 1

    def section_index_at_time(self, time_ms: float) -> int:
        time_value_ms = _require_finite_float(time_ms, "time_ms")
        if time_value_ms < self.start_time_ms or time_value_ms > self.end_time_ms:
            raise ValueError("time_ms is outside the closed curve time domain")
        if time_value_ms == self.end_time_ms:
            return len(self.sections) - 1
        return bisect.bisect_right(self._boundary_times_ms, time_value_ms) - 1

    def section_at_beat(self, beat: float) -> TimingCurveSection:
        return self.sections[self.section_index_at_beat(beat)]

    def section_at_time(self, time_ms: float) -> TimingCurveSection:
        return self.sections[self.section_index_at_time(time_ms)]

    def time_at_beat(self, beat: float) -> float:
        beat_value = _require_finite_real(beat, "beat")
        section_index = self.section_index_at_beat(beat_value)
        if beat_value == self.end_beat:
            return self.end_time_ms
        if beat_value == self.sections[section_index].start_beat:
            return self._boundary_times_ms[section_index]

        section = self.sections[section_index]
        if isinstance(section, ConstantTempoSection):
            elapsed_seconds = 60.0 * (beat_value - section.start_beat) / section.bpm
        else:
            elapsed_seconds = section.elapsed_seconds_at_beat(beat_value)
        result = self._boundary_times_ms[section_index] + 1000.0 * elapsed_seconds
        if not math.isfinite(result):
            raise ValueError("derived time must be finite")
        return result

    def beat_at_time(self, time_ms: float) -> float:
        time_value_ms = _require_finite_float(time_ms, "time_ms")
        section_index = self.section_index_at_time(time_value_ms)
        if time_value_ms == self.end_time_ms:
            return float(self.end_beat)
        if time_value_ms == self._boundary_times_ms[section_index]:
            return float(self.sections[section_index].start_beat)

        section = self.sections[section_index]
        elapsed_seconds = (time_value_ms - self._boundary_times_ms[section_index]) / 1000.0
        if isinstance(section, ConstantTempoSection):
            beat = section.start_beat + section.bpm * elapsed_seconds / 60.0
        else:
            beat = section.beat_at_elapsed_seconds(elapsed_seconds)
        if not math.isfinite(beat):
            raise ValueError("derived beat must be finite")
        return beat

    def bpm_at_time(self, time_ms: float) -> float:
        time_value_ms = _require_finite_float(time_ms, "time_ms")
        section_index = self.section_index_at_time(time_value_ms)
        section = self.sections[section_index]
        if time_value_ms == self.end_time_ms:
            return section.end_bpm
        elapsed_seconds = (time_value_ms - self._boundary_times_ms[section_index]) / 1000.0
        if isinstance(section, ConstantTempoSection):
            return section.bpm
        return section.bpm_at_elapsed_seconds(elapsed_seconds)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": ANALYTIC_CURVE_VERSION,
            "origin_beat": self.origin_beat,
            "origin_time_ms": _canonical_float(self.origin_time_ms),
            "sections": [section.to_dict() for section in self.sections],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PhaseContinuousTimingCurve:
        _require_exact_keys(
            payload,
            {"version", "origin_beat", "origin_time_ms", "sections"},
            "curve",
        )
        if payload["version"] != ANALYTIC_CURVE_VERSION:
            raise ValueError(f"version must be exactly {ANALYTIC_CURVE_VERSION!r}")
        sections_payload = payload["sections"]
        if not isinstance(sections_payload, Sequence) or isinstance(
            sections_payload, (str, bytes, bytearray)
        ):
            raise ValueError("sections must be a sequence")
        sections = tuple(
            _section_from_dict(section_payload, index)
            for index, section_payload in enumerate(sections_payload)
        )
        return cls(
            origin_beat=payload["origin_beat"],
            origin_time_ms=_parse_canonical_float(payload["origin_time_ms"], "origin_time_ms"),
            sections=sections,
        )

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")

    def to_canonical_bytes(self) -> bytes:
        return self.canonical_bytes()

    @property
    def fingerprint_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @property
    def canonical_fingerprint_sha256(self) -> str:
        return self.fingerprint_sha256

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> PhaseContinuousTimingCurve:
        if not isinstance(payload, bytes):
            raise ValueError("canonical curve payload must be bytes")
        try:
            decoded = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("canonical curve payload must be valid UTF-8") from exc
        try:
            parsed = json.loads(
                decoded,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("canonical curve payload must be valid strict JSON") from exc
        curve = cls.from_dict(parsed)
        if curve.canonical_bytes() != payload:
            raise ValueError("curve payload is not in exact canonical JSON byte form")
        return curve

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> PhaseContinuousTimingCurve:
        return cls.from_canonical_bytes(payload)


def solve_linear_time_ramp_elapsed_seconds(
    beat_offset: float,
    *,
    start_bpm: float,
    acceleration_bpm_per_second: float,
) -> float:
    """Invert integrated time-linear BPM with the frozen stable expression."""

    beat_offset_value = _require_finite_float(beat_offset, "beat_offset")
    start_bpm_value = _require_bpm(start_bpm, "start_bpm")
    acceleration = _require_finite_float(
        acceleration_bpm_per_second,
        "acceleration_bpm_per_second",
    )
    if beat_offset_value < 0.0:
        raise ValueError("beat_offset must be nonnegative")
    if acceleration == 0.0:
        # This is an exact-zero case, not a near-zero tolerance branch.  Keeping
        # the constant formula explicit also gives it identical float64
        # semantics to ConstantTempoSection for the Exp009 oracle.
        return 60.0 * beat_offset_value / start_bpm_value
    radicand = start_bpm_value * start_bpm_value + 120.0 * acceleration * beat_offset_value
    if not math.isfinite(radicand) or radicand < 0.0:
        raise ValueError("ramp inverse radicand must be finite and nonnegative")
    denominator = start_bpm_value + math.sqrt(radicand)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("ramp inverse denominator must be finite and positive")
    elapsed = 120.0 * beat_offset_value / denominator
    if not math.isfinite(elapsed):
        raise ValueError("derived ramp elapsed time must be finite")
    return elapsed


def _section_from_dict(payload: object, index: int) -> TimingCurveSection:
    if not isinstance(payload, Mapping):
        raise ValueError(f"sections[{index}] must be a mapping")
    section_type = payload.get("type")
    if section_type == "constant":
        _require_exact_keys(payload, {"type", "start_beat", "end_beat", "bpm"}, f"sections[{index}]")
        return ConstantTempoSection(
            start_beat=payload["start_beat"],
            end_beat=payload["end_beat"],
            bpm=_parse_canonical_float(payload["bpm"], f"sections[{index}].bpm"),
        )
    if section_type == "linear_bpm_time":
        _require_exact_keys(
            payload,
            {"type", "start_beat", "end_beat", "start_bpm", "end_bpm"},
            f"sections[{index}]",
        )
        return LinearTimeRampSection(
            start_beat=payload["start_beat"],
            end_beat=payload["end_beat"],
            start_bpm=_parse_canonical_float(
                payload["start_bpm"],
                f"sections[{index}].start_bpm",
            ),
            end_bpm=_parse_canonical_float(
                payload["end_bpm"],
                f"sections[{index}].end_bpm",
            ),
        )
    raise ValueError(f"sections[{index}].type must be 'constant' or 'linear_bpm_time'")


def _require_section(value: object, index: int) -> TimingCurveSection:
    if not isinstance(value, (ConstantTempoSection, LinearTimeRampSection)):
        raise ValueError(
            f"sections[{index}] must be a ConstantTempoSection or LinearTimeRampSection"
        )
    return value


def _constant_duration_seconds(beat_count: int, bpm: float) -> float:
    try:
        duration = 60.0 * beat_count / bpm
    except OverflowError as exc:
        raise ValueError("derived constant-section duration must be finite") from exc
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("derived constant-section duration must be finite and positive")
    return duration


def _ramp_duration_seconds(beat_count: int, start_bpm: float, end_bpm: float) -> float:
    try:
        duration = 120.0 * beat_count / (start_bpm + end_bpm)
    except OverflowError as exc:
        raise ValueError("derived ramp duration must be finite") from exc
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("derived ramp duration must be finite and positive")
    return duration


def _clamp_ramp_elapsed_seconds(value: float, duration_seconds: float) -> float:
    if value < 0.0:
        if value >= -_RAMP_ENDPOINT_TOLERANCE_SECONDS:
            return 0.0
        raise ValueError("derived ramp elapsed time is before the section start")
    if value > duration_seconds:
        if value <= duration_seconds + _RAMP_ENDPOINT_TOLERANCE_SECONDS:
            return duration_seconds
        raise ValueError("derived ramp elapsed time is after the section end")
    return value


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
        raise ValueError(
            f"{name} must be within [{TIMING_V3_MIN_BPM:g}, {TIMING_V3_MAX_BPM:g}]"
        )
    return bpm


def _canonical_float(value: float) -> str:
    return format(float(value), ".17g")


def _parse_canonical_float(value: object, name: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a canonical float string")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a canonical float string") from exc
    if not math.isfinite(parsed) or _canonical_float(parsed) != value:
        raise ValueError(f"{name} must be a canonical finite float string")
    return parsed


def _require_exact_keys(payload: object, expected: set[str], name: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a mapping")
    keys = set(payload.keys())
    if keys != expected:
        raise ValueError(f"{name} must have exactly keys {sorted(expected)!r}")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


__all__ = [
    "ANALYTIC_CURVE_VERSION",
    "TIMING_V3_ANALYTIC_CURVE_VERSION",
    "ConstantTempoSection",
    "LinearTimeRampSection",
    "PhaseContinuousTimingCurve",
    "TimingCurveSeam",
    "TimingCurveSection",
    "solve_linear_time_ramp_elapsed_seconds",
]
