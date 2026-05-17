from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

T = TypeVar("T")


def definitely_bigger(value1: float, value2: float, acceptable_difference: float = 1.0) -> bool:
    return value1 - acceptable_difference > value2


def logistic(x: float, midpoint_offset: float, multiplier: float, max_value: float = 1.0) -> float:
    return max_value / (1.0 + math.exp(multiplier * (midpoint_offset - x)))


def clamp(value: int, lo: int, hi: int) -> int:
    return lo if value < lo else hi if value > hi else value


def legacy_sort_in_place(items: list[T], comparer: Callable[[T, T], int]) -> None:
    if not items:
        return
    _depth_limited_quick_sort(items, 0, len(items) - 1, comparer, 32)


def _swap_if_greater(keys: list[T], comparer: Callable[[T, T], int], a: int, b: int) -> None:
    if a != b and comparer(keys[a], keys[b]) > 0:
        keys[a], keys[b] = keys[b], keys[a]


def _swap(keys: list[T], i: int, j: int) -> None:
    if i != j:
        keys[i], keys[j] = keys[j], keys[i]


def _down_heap(keys: list[T], i: int, n: int, lo: int, comparer: Callable[[T, T], int]) -> None:
    d = keys[lo + i - 1]
    while i <= n // 2:
        child = 2 * i
        if child < n and comparer(keys[lo + child - 1], keys[lo + child]) < 0:
            child += 1
        if not (comparer(d, keys[lo + child - 1]) < 0):
            break
        keys[lo + i - 1] = keys[lo + child - 1]
        i = child
    keys[lo + i - 1] = d


def _heapsort(keys: list[T], lo: int, hi: int, comparer: Callable[[T, T], int]) -> None:
    n = hi - lo + 1
    i = n // 2
    while i >= 1:
        _down_heap(keys, i, n, lo, comparer)
        i -= 1
    i = n
    while i > 1:
        _swap(keys, lo, lo + i - 1)
        _down_heap(keys, 1, i - 1, lo, comparer)
        i -= 1


def _depth_limited_quick_sort(
    keys: list[T],
    left: int,
    right: int,
    comparer: Callable[[T, T], int],
    depth_limit: int,
) -> None:
    while left < right:
        if depth_limit == 0:
            _heapsort(keys, left, right, comparer)
            return

        i = left
        j = right

        middle = i + ((j - i) >> 1)
        _swap_if_greater(keys, comparer, i, middle)
        _swap_if_greater(keys, comparer, i, j)
        _swap_if_greater(keys, comparer, middle, j)

        x = keys[middle]

        while i <= j:
            while comparer(keys[i], x) < 0:
                i += 1
            while comparer(x, keys[j]) < 0:
                j -= 1
            if i > j:
                break
            if i < j:
                keys[i], keys[j] = keys[j], keys[i]
            i += 1
            j -= 1

        depth_limit -= 1

        if j - left <= right - i:
            if left < j:
                _depth_limited_quick_sort(keys, left, j, comparer, depth_limit)
            left = i
        else:
            if i < right:
                _depth_limited_quick_sort(keys, i, right, comparer, depth_limit)
            right = j


@dataclass(frozen=True)
class RawHitObject:
    start_time: float
    end_time: float
    column: int


@dataclass(frozen=True)
class ParsedOsu:
    path: Path
    audio_filename: str
    mode: int
    circle_size: float
    difficulty_name: str
    hit_objects: list[RawHitObject]


def _validate_audio_matches(parsed: ParsedOsu, osu_path: Path, audio_path: Path) -> None:
    expected_audio_path = (osu_path.parent / parsed.audio_filename).resolve()
    actual_audio_path = audio_path.resolve()

    if expected_audio_path.exists():
        if expected_audio_path != actual_audio_path:
            raise ValueError(
                f'AudioFilename mismatch: .osu expects "{parsed.audio_filename}" at "{expected_audio_path}", got "{audio_path}"',
            )
    elif Path(parsed.audio_filename).name.casefold() != audio_path.name.casefold():
        raise ValueError(
            f'AudioFilename mismatch: .osu expects "{parsed.audio_filename}", got "{audio_path.name}"',
        )


def parse_osu_file(path: str | Path) -> ParsedOsu:
    path = Path(path)
    audio_filename = ""
    mode: int | None = None
    circle_size: float | None = None
    difficulty_name = ""
    in_hit_objects = False
    hit_objects: list[RawHitObject] = []

    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue

            if line.startswith("[") and line.endswith("]"):
                in_hit_objects = line == "[HitObjects]"
                continue

            if not in_hit_objects:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                if key == "AudioFilename":
                    audio_filename = value
                elif key == "Mode":
                    try:
                        mode = int(value)
                    except ValueError:
                        pass
                elif key == "CircleSize":
                    try:
                        circle_size = float(value)
                    except ValueError:
                        pass
                elif key == "Version":
                    difficulty_name = value
                continue

            parts = line.split(",")
            if len(parts) < 5:
                continue

            try:
                x = int(parts[0])
                start_time = float(parts[2])
                obj_type = int(parts[3])
            except ValueError:
                continue

            if mode != 3:
                continue
            if circle_size is None:
                raise ValueError(f"{path} is missing CircleSize before [HitObjects].")

            total_columns = max(1, int(round(circle_size)))
            column = clamp(int(math.floor(x * total_columns / 512.0)), 0, total_columns - 1)

            is_hold = (obj_type & 128) != 0
            is_circle = (obj_type & 1) != 0

            if is_hold:
                if len(parts) < 6:
                    raise ValueError(f"Malformed mania hold note in {path}: {line}")
                end_field = parts[5]
                end_time_str = end_field.split(":", 1)[0]
                try:
                    end_time = float(end_time_str)
                except ValueError as exc:
                    raise ValueError(f"Malformed mania hold end time in {path}: {line}") from exc
                hit_objects.append(RawHitObject(start_time=start_time, end_time=end_time, column=column))
            elif is_circle:
                hit_objects.append(RawHitObject(start_time=start_time, end_time=start_time, column=column))
            else:
                raise ValueError(f"Unsupported hit object type for mania difficulty in {path}: type={obj_type}, line={line}")

    if mode is None:
        raise ValueError(f"{path} is missing Mode.")
    if not audio_filename:
        raise ValueError(f"{path} is missing AudioFilename.")
    if circle_size is None:
        raise ValueError(f"{path} is missing CircleSize.")
    if mode != 3:
        raise ValueError(f"{path} is not an osu!mania map (Mode={mode}).")

    return ParsedOsu(
        path=path,
        audio_filename=audio_filename,
        mode=mode,
        circle_size=circle_size,
        difficulty_name=difficulty_name or path.stem,
        hit_objects=hit_objects,
    )


class DifficultyHitObject:
    def __init__(
        self,
        base_object: RawHitObject,
        last_object: RawHitObject,
        clock_rate: float,
        objects: Sequence["DifficultyHitObject"],
        index: int,
    ) -> None:
        self._difficulty_hit_objects = objects
        self.index = index
        self.base_object = base_object
        self.last_object = last_object
        self.delta_time = (base_object.start_time - last_object.start_time) / clock_rate
        self.start_time = base_object.start_time / clock_rate
        self.end_time = base_object.end_time / clock_rate

    def previous(self, backwards_index: int) -> "DifficultyHitObject | None":
        idx = self.index - (backwards_index + 1)
        if 0 <= idx < len(self._difficulty_hit_objects):
            return self._difficulty_hit_objects[idx]
        return None


class ManiaDifficultyHitObject(DifficultyHitObject):
    def __init__(
        self,
        hit_object: RawHitObject,
        last_object: RawHitObject,
        clock_rate: float,
        objects: Sequence[DifficultyHitObject],
        per_column_objects: list[list["ManiaDifficultyHitObject"]],
        index: int,
    ) -> None:
        super().__init__(hit_object, last_object, clock_rate, objects, index)
        total_columns = len(per_column_objects)
        self._per_column_objects = per_column_objects
        self.column = hit_object.column
        self._column_index = len(per_column_objects[self.column])
        self.previous_hit_objects: list[ManiaDifficultyHitObject | None] = [None] * total_columns

        prev_in_column = self.prev_in_column(0)
        self.column_strain_time = self.start_time - prev_in_column.start_time if prev_in_column is not None else self.start_time

        if index > 0:
            prev_note = objects[index - 1]
            assert isinstance(prev_note, ManiaDifficultyHitObject)
            for i in range(len(prev_note.previous_hit_objects)):
                self.previous_hit_objects[i] = prev_note.previous_hit_objects[i]
            self.previous_hit_objects[prev_note.column] = prev_note

    def prev_in_column(self, backwards_index: int) -> "ManiaDifficultyHitObject | None":
        idx = self._column_index - (backwards_index + 1)
        column_objects = self._per_column_objects[self.column]
        if 0 <= idx < len(column_objects):
            return column_objects[idx]
        return None


class Skill:
    def process(self, current: DifficultyHitObject) -> None:
        raise NotImplementedError

    def difficulty_value(self) -> float:
        raise NotImplementedError


class StrainSkill(Skill):
    decay_weight = 0.9
    section_length = 400

    def __init__(self) -> None:
        self.current_section_peak = 0.0
        self.current_section_end = 0.0
        self.strain_peaks: list[float] = []
        self.object_strains: list[float] = []

    def strain_value_at(self, current: DifficultyHitObject) -> float:
        raise NotImplementedError

    def calculate_initial_strain(self, time: float, current: DifficultyHitObject) -> float:
        raise NotImplementedError

    def process(self, current: DifficultyHitObject) -> None:
        if current.index == 0:
            self.current_section_end = math.ceil(current.start_time / self.section_length) * self.section_length

        while current.start_time > self.current_section_end:
            self._save_current_peak()
            self._start_new_section_from(self.current_section_end, current)
            self.current_section_end += self.section_length

        strain = self.strain_value_at(current)
        self.current_section_peak = max(strain, self.current_section_peak)
        self.object_strains.append(strain)

    def _save_current_peak(self) -> None:
        self.strain_peaks.append(self.current_section_peak)

    def _start_new_section_from(self, time: float, current: DifficultyHitObject) -> None:
        self.current_section_peak = self.calculate_initial_strain(time, current)

    def get_current_strain_peaks(self) -> list[float]:
        return [*self.strain_peaks, self.current_section_peak]

    def difficulty_value(self) -> float:
        difficulty = 0.0
        weight = 1.0
        peaks = [peak for peak in self.get_current_strain_peaks() if peak > 0]
        for strain in sorted(peaks, reverse=True):
            difficulty += strain * weight
            weight *= self.decay_weight
        return difficulty


class StrainDecaySkill(StrainSkill):
    skill_multiplier = 1.0
    strain_decay_base = 1.0

    def __init__(self) -> None:
        super().__init__()
        self.current_strain = 0.0

    def _strain_decay(self, ms: float) -> float:
        return math.pow(self.strain_decay_base, ms / 1000.0)

    def calculate_initial_strain(self, time: float, current: DifficultyHitObject) -> float:
        prev = current.previous(0)
        if prev is None:
            return 0.0
        return self.current_strain * self._strain_decay(time - prev.start_time)

    def strain_value_of(self, current: DifficultyHitObject) -> float:
        raise NotImplementedError

    def strain_value_at(self, current: DifficultyHitObject) -> float:
        self.current_strain *= self._strain_decay(current.delta_time)
        self.current_strain += self.strain_value_of(current) * self.skill_multiplier
        return self.current_strain


def individual_strain_evaluate(current: DifficultyHitObject) -> float:
    mania_current = current
    assert isinstance(mania_current, ManiaDifficultyHitObject)

    start_time = mania_current.start_time
    end_time = mania_current.end_time
    hold_factor = 1.0

    for mania_previous in mania_current.previous_hit_objects:
        if mania_previous is None:
            continue
        if definitely_bigger(mania_previous.end_time, end_time, 1) and definitely_bigger(start_time, mania_previous.start_time, 1):
            hold_factor = 1.25
            break

    return 2.0 * hold_factor


def overall_strain_evaluate(current: DifficultyHitObject) -> float:
    mania_current = current
    assert isinstance(mania_current, ManiaDifficultyHitObject)

    release_threshold = 30.0
    start_time = mania_current.start_time
    end_time = mania_current.end_time

    is_overlapping = False
    closest_end_time = abs(end_time - start_time)
    hold_factor = 1.0
    hold_addition = 0.0

    for mania_previous in mania_current.previous_hit_objects:
        if mania_previous is None:
            continue

        is_overlapping = is_overlapping or (
            definitely_bigger(mania_previous.end_time, start_time, 1)
            and definitely_bigger(end_time, mania_previous.end_time, 1)
            and definitely_bigger(start_time, mania_previous.start_time, 1)
        )

        if definitely_bigger(mania_previous.end_time, end_time, 1) and definitely_bigger(start_time, mania_previous.start_time, 1):
            hold_factor = 1.25

        closest_end_time = min(closest_end_time, abs(end_time - mania_previous.end_time))

    if is_overlapping:
        hold_addition = logistic(x=closest_end_time, multiplier=0.27, midpoint_offset=release_threshold)

    return (1.0 + hold_addition) * hold_factor


class ManiaStrain(StrainDecaySkill):
    individual_decay_base = 0.125
    overall_decay_base = 0.30

    skill_multiplier = 1.0
    strain_decay_base = 1.0

    def __init__(self, total_columns: int) -> None:
        super().__init__()
        self.individual_strains = [0.0] * total_columns
        self.highest_individual_strain = 0.0
        self.overall_strain = 1.0

    @staticmethod
    def _apply_decay(value: float, delta_time: float, decay_base: float) -> float:
        return value * math.pow(decay_base, delta_time / 1000.0)

    def strain_value_of(self, current: DifficultyHitObject) -> float:
        mania_current = current
        assert isinstance(mania_current, ManiaDifficultyHitObject)

        col = mania_current.column
        self.individual_strains[col] = self._apply_decay(
            self.individual_strains[col],
            mania_current.column_strain_time,
            self.individual_decay_base,
        )
        self.individual_strains[col] += individual_strain_evaluate(current)

        if mania_current.delta_time <= 1:
            self.highest_individual_strain = max(self.highest_individual_strain, self.individual_strains[col])
        else:
            self.highest_individual_strain = self.individual_strains[col]

        self.overall_strain = self._apply_decay(self.overall_strain, mania_current.delta_time, self.overall_decay_base)
        self.overall_strain += overall_strain_evaluate(current)

        return self.highest_individual_strain + self.overall_strain - self.current_strain

    def calculate_initial_strain(self, time: float, current: DifficultyHitObject) -> float:
        prev = current.previous(0)
        if prev is None:
            return 0.0
        delta = time - prev.start_time
        return self._apply_decay(self.highest_individual_strain, delta, self.individual_decay_base) + self._apply_decay(
            self.overall_strain,
            delta,
            self.overall_decay_base,
        )


def create_difficulty_hit_objects(
    raw_objects: list[RawHitObject],
    total_columns: int,
    clock_rate: float,
) -> list[ManiaDifficultyHitObject]:
    sorted_objects = list(raw_objects)

    def comparer(a: RawHitObject, b: RawHitObject) -> int:
        return int(round(a.start_time)) - int(round(b.start_time))

    legacy_sort_in_place(sorted_objects, comparer)

    objects: list[ManiaDifficultyHitObject] = []
    per_column_objects: list[list[ManiaDifficultyHitObject]] = [[] for _ in range(total_columns)]

    for i in range(1, len(sorted_objects)):
        current = ManiaDifficultyHitObject(
            hit_object=sorted_objects[i],
            last_object=sorted_objects[i - 1],
            clock_rate=clock_rate,
            objects=objects,
            per_column_objects=per_column_objects,
            index=len(objects),
        )
        objects.append(current)
        per_column_objects[current.column].append(current)

    return objects


def compute_mania_star_rating_20241007(
    raw_objects: list[RawHitObject],
    total_columns: int,
    clock_rate: float,
) -> float:
    difficulty_multiplier = 0.018
    diff_objects = create_difficulty_hit_objects(raw_objects, total_columns, clock_rate=clock_rate)
    strain = ManiaStrain(total_columns=total_columns)
    for obj in diff_objects:
        strain.process(obj)
    return strain.difficulty_value() * difficulty_multiplier


def calculate_mania_difficulties(
    osu_path: str | Path,
    audio_path: str | Path,
    speeds: Iterable[float],
) -> list[float]:
    osu_path = Path(osu_path)
    audio_path = Path(audio_path)
    speed_list = list(speeds)

    for speed in speed_list:
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError(f"speed must be positive, got {speed}")
    if not osu_path.is_file():
        raise FileNotFoundError(f"osu file not found: {osu_path}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    parsed = parse_osu_file(osu_path)
    _validate_audio_matches(parsed, osu_path, audio_path)

    total_columns = max(1, int(round(parsed.circle_size)))
    return [
        compute_mania_star_rating_20241007(parsed.hit_objects, total_columns, clock_rate=speed)
        for speed in speed_list
    ]


def calculate_mania_difficulty(osu_path: str | Path, audio_path: str | Path, speed: float) -> float:
    return calculate_mania_difficulties(osu_path, audio_path, [speed])[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print official 20241007 osu!mania difficulty as a single .2f float.")
    parser.add_argument("osu", type=Path, help="Path to the osu!mania .osu file.")
    parser.add_argument("audio", type=Path, help="Path to the chart audio file. Must match AudioFilename in the .osu.")
    parser.add_argument("--speed", type=float, default=1.0, help="Clock-rate multiplier, e.g. 1.0, 1.25, 1.5.")
    args = parser.parse_args(argv)

    try:
        difficulty = calculate_mania_difficulty(args.osu, args.audio, speed=args.speed)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"{difficulty:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
