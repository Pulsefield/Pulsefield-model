import pytest

from pulsefield_model.research.oracle_time_continuation.data import admit_source
from pulsefield_model.research.scoped_style_modeling.dataset import digest


def source_bytes(objects):
    lines = ["osu file format v14", "[General]", "Mode:3", "[Difficulty]", "CircleSize:4", "[HitObjects]"]
    for lane, start, end in objects:
        kind = 128 if end > start else 1
        tail = f"{end}:0:0:0:0:" if kind == 128 else "0:0:0:0:"
        lines.append(f"{64 + 128 * lane},192,{start},{kind},0,{tail}")
    return ("\n".join(lines) + "\n").encode()


def admit(objects):
    data = source_bytes(objects)
    return admit_source(data, digest(data), group_id="song:fixture", split="train")


@pytest.fixture
def chord_source():
    # 28 notes in seven chords, one tap, then a four-note threshold row: 33 notes.
    objects = [(lane, i * 100, 50 if (i, lane) == (0, 0) else i * 100)
               for i in range(7) for lane in range(4)]
    objects += [(0, 700, 700)]
    objects += [(lane, 800, 900 + lane * 100) for lane in range(4)]
    objects += [(0, 1300, 1300)]
    return admit(objects)
