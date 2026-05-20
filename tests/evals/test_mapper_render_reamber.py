from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pulsefield_model.evals import mapper_render_reamber as render_reamber


def _write_osu(path: Path, hitobject_lines: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "osu file format v14",
                "",
                "[General]",
                "AudioFilename: song.mp3",
                "Mode:3",
                "",
                "[Metadata]",
                "Title:Render Test",
                "Artist:Pulsefield",
                "Creator:tests",
                "Version:Generated",
                "",
                "[Difficulty]",
                "CircleSize:4",
                "OverallDifficulty:5",
                "",
                "[TimingPoints]",
                "0,500,4,2,0,100,1,0",
                "",
                "[HitObjects]",
                *hitobject_lines,
            ],
        ),
        encoding="utf-8",
    )


def _sample_chart(path: Path) -> None:
    _write_osu(
        path,
        [
            "64,192,0,1,0,0:0:0:0:",
            "192,192,1000,1,0,0:0:0:0:",
            "320,192,20000,1,0,0:0:0:0:",
            "448,192,60000,1,0,0:0:0:0:",
            "64,192,65000,1,0,0:0:0:0:",
            "64,192,66000,1,0,0:0:0:0:",
            "64,192,67000,1,0,0:0:0:0:",
            "64,192,68000,1,0,0:0:0:0:",
            "448,192,90000,1,0,0:0:0:0:",
        ],
    )


def test_plan_named_spans_selects_required_spans(tmp_path: Path) -> None:
    osu_path = tmp_path / "generated.osu"
    _sample_chart(osu_path)

    spans = render_reamber.plan_named_spans(osu_path)

    assert tuple(spans) == render_reamber.SPAN_NAMES
    assert (spans["first_30s"].start_ms, spans["first_30s"].end_ms) == (0.0, 30000.0)
    assert (spans["middle_30s"].start_ms, spans["middle_30s"].end_ms) == (30000.0, 60000.0)
    assert (spans["last_30s"].start_ms, spans["last_30s"].end_ms) == (60000.0, 90000.0)
    assert (spans["longest_empty_span"].start_ms, spans["longest_empty_span"].end_ms) == (
        25000.0,
        55000.0,
    )
    assert spans["most_repetitive_span"].start_ms == 45000.0


def test_reamber_available_returns_bool() -> None:
    assert isinstance(render_reamber.reamber_available(), bool)


def test_render_raises_clear_runtime_error_when_reamber_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def missing_reamber(name: str):
        if name.startswith("reamber"):
            raise ImportError(name)
        return original_import_module(name)

    original_import_module = render_reamber.importlib.import_module
    monkeypatch.setattr(render_reamber.importlib, "import_module", missing_reamber)

    with pytest.raises(RuntimeError, match="optional dependency 'reamber'.*render extra"):
        render_reamber.render_named_spans(tmp_path / "missing.osu", tmp_path)


def test_render_uses_reamber_api_and_writes_named_pngs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image = pytest.importorskip("PIL.Image")
    osu_path = tmp_path / "generated.osu"
    _sample_chart(osu_path)
    layers: list[str] = []
    read_paths: list[Path] = []

    class FakeOsuMap:
        @staticmethod
        def read_file(path: str | Path) -> object:
            read_paths.append(Path(path))
            return object()

    class FakePlayField:
        def __init__(self, *, m: object, duration_per_px: float, padding: int) -> None:
            self.m = m
            self.duration_per_px = duration_per_px
            self.padding = padding
            self.start = 0.0
            self.canvas_h = 100
            self.hit_height = 0
            self.background_color = "#000000"

        def __add__(self, other: object) -> "FakePlayField":
            layers.append(type(other).__name__)
            return self

        def export(self):
            return image.new("RGB", (20, self.canvas_h), color="#111111")

    class FakeDraw:
        pass

    fake_api = SimpleNamespace(
        OsuMap=FakeOsuMap,
        PlayField=FakePlayField,
        PFDrawBpm=type("PFDrawBpm", (FakeDraw,), {}),
        PFDrawBeatLines=type("PFDrawBeatLines", (FakeDraw,), {}),
        PFDrawColumnLines=type("PFDrawColumnLines", (FakeDraw,), {}),
        PFDrawNotes=type("PFDrawNotes", (FakeDraw,), {}),
        PFDrawOffsets=type("PFDrawOffsets", (FakeDraw,), {}),
    )
    monkeypatch.setattr(render_reamber, "_load_reamber_api", lambda: fake_api)

    paths = render_reamber.render_named_spans(
        osu_path,
        tmp_path / "renders",
        duration_per_px=1000,
        fold_max_height=None,
    )

    assert tuple(paths) == render_reamber.SPAN_NAMES
    assert read_paths == [osu_path]
    assert layers == [
        "PFDrawBpm",
        "PFDrawBeatLines",
        "PFDrawColumnLines",
        "PFDrawNotes",
        "PFDrawOffsets",
    ]
    for name, path in paths.items():
        assert path.name == f"generated__{name}.png"
        assert path.exists()


def test_real_reamber_render_if_installed(tmp_path: Path) -> None:
    pytest.importorskip("reamber")
    pytest.importorskip("PIL.Image")
    osu_path = tmp_path / "generated.osu"
    _sample_chart(osu_path)

    paths = render_reamber.render_named_spans(
        osu_path,
        tmp_path / "renders",
        duration_per_px=1000,
        fold_max_height=None,
    )

    assert tuple(paths) == render_reamber.SPAN_NAMES
    assert all(path.exists() for path in paths.values())
