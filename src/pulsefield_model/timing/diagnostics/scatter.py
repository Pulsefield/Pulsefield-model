from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ScatterSpec:
    name: str
    x_metric: str
    y_metric: str
    x_label: str
    y_label: str
    title: str
    optional: bool = False


@dataclass(frozen=True)
class ScatterPoints:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    sample_indexes: list[int | None]


DEFAULT_SCATTER_SPECS: tuple[ScatterSpec, ...] = (
    ScatterSpec(
        name="fit_seconds_by_candidate_count",
        x_metric="candidate_count",
        y_metric="fit_seconds",
        x_label="Grid candidates evaluated",
        y_label="Fit seconds",
        title="GridFitter runtime by candidate count",
    ),
    ScatterSpec(
        name="fit_seconds_by_frame_count",
        x_metric="frame_count",
        y_metric="fit_seconds",
        x_label="Frame count",
        y_label="Fit seconds",
        title="GridFitter runtime by frame count",
    ),
    ScatterSpec(
        name="mean_phase_error_ms_by_segment_count_delta",
        x_metric="segment_count_delta",
        y_metric="mean_phase_error_ms",
        x_label="Predicted segment count - oracle segment count",
        y_label="Mean phase error (ms)",
        title="Phase error by segment-count delta",
    ),
    ScatterSpec(
        name="local_bpm_mae_by_tempo_multiplier",
        x_metric="tempo_multiplier",
        y_metric="local_bpm_mae",
        x_label="Selected tempo multiplier",
        y_label="Local BPM MAE",
        title="Local BPM error by tempo multiplier",
    ),
    ScatterSpec(
        name="local_bpm_alias_mae_by_tempo_multiplier",
        x_metric="tempo_multiplier",
        y_metric="local_bpm_alias_mae",
        x_label="Selected tempo multiplier",
        y_label="Alias-aware local BPM MAE",
        title="Alias-aware local BPM error by tempo multiplier",
        optional=True,
    ),
)


def load_report_json(report_json_path: Path) -> dict[str, object]:
    with report_json_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError(f"report JSON must contain an object, got {type(report).__name__}")
    return report


def write_scatter_artifacts(
    report: Mapping[str, object],
    *,
    output_dir: Path,
    source_report_path: Path | None = None,
    specs: Sequence[ScatterSpec] = DEFAULT_SCATTER_SPECS,
    dpi: int = 150,
) -> dict[str, object]:
    rows = _report_rows(report)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots: list[dict[str, object]] = []
    skipped_plots: list[dict[str, object]] = []
    for spec in specs:
        missing_metrics = _missing_metrics(rows, spec)
        if missing_metrics and spec.optional:
            skipped_plots.append(
                {
                    "name": spec.name,
                    "reason": "missing metric",
                    "missing_metrics": missing_metrics,
                }
            )
            continue
        points = _scatter_points(rows, spec)
        if len(points.x) == 0:
            raise ValueError(f"scatter plot {spec.name!r} has no finite points")

        output_path = output_dir / f"{spec.name}.png"
        _write_scatter_plot(points, spec, output_path=output_path, dpi=dpi)
        plots.append(
            {
                "name": spec.name,
                "path": output_path.name,
                "point_count": int(len(points.x)),
                "x_metric": spec.x_metric,
                "y_metric": spec.y_metric,
                "x_min": float(np.min(points.x)),
                "x_max": float(np.max(points.x)),
                "y_min": float(np.min(points.y)),
                "y_max": float(np.max(points.y)),
                "pearson_r": _pearson_r(points.x, points.y),
            }
        )

    manifest: dict[str, object] = {
        "source_report_path": None if source_report_path is None else source_report_path.as_posix(),
        "plots": plots,
        "skipped_plots": skipped_plots,
        "specs": [asdict(spec) for spec in specs],
    }
    manifest_path = output_dir / "diagnostics_scatter_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render scatter diagnostics from a stage2 timing audit JSON report.")
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)

    report = load_report_json(args.report_json)
    manifest = write_scatter_artifacts(
        report,
        output_dir=args.output_dir,
        source_report_path=args.report_json,
        dpi=args.dpi,
    )
    print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))
    return 0


def _report_rows(report: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ValueError("report must contain a list at key 'rows'")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"report row {index} must be an object")
    return rows


def _scatter_points(
    rows: Sequence[Mapping[str, object]],
    spec: ScatterSpec,
) -> ScatterPoints:
    x_values: list[float] = []
    y_values: list[float] = []
    sample_indexes: list[int | None] = []

    for row_index, row in enumerate(rows):
        if spec.x_metric not in row or spec.y_metric not in row:
            raise ValueError(
                "missing required scatter metric "
                f"{spec.x_metric!r} or {spec.y_metric!r} in row {row_index}"
            )
        x = _finite_float_or_none(row[spec.x_metric])
        y = _finite_float_or_none(row[spec.y_metric])
        if x is None or y is None:
            continue
        x_values.append(x)
        y_values.append(y)
        sample_indexes.append(_sample_index(row))

    return ScatterPoints(
        x=np.asarray(x_values, dtype=np.float64),
        y=np.asarray(y_values, dtype=np.float64),
        sample_indexes=sample_indexes,
    )


def _missing_metrics(rows: Sequence[Mapping[str, object]], spec: ScatterSpec) -> list[str]:
    required_metrics = (spec.x_metric, spec.y_metric)
    missing: list[str] = []
    for metric in required_metrics:
        if any(metric not in row for row in rows):
            missing.append(metric)
    return missing


def _write_scatter_plot(
    points: ScatterPoints,
    spec: ScatterSpec,
    *,
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    ax.scatter(points.x, points.y, s=18, alpha=0.65, linewidths=0.0, color="#2f6f9f")
    ax.set_title(spec.title)
    ax.set_xlabel(spec.x_label)
    ax.set_ylabel(spec.y_label)
    ax.grid(True, color="#d7dde3", linewidth=0.8, alpha=0.8)
    _add_percentile_guides(ax, points.y)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _add_percentile_guides(ax: plt.Axes, values: NDArray[np.float64]) -> None:
    if len(values) < 2:
        return
    p95 = float(np.percentile(values, 95.0))
    ax.axhline(p95, color="#ba3b46", linewidth=1.0, linestyle="--", alpha=0.75)
    ax.text(
        0.99,
        p95,
        "p95",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        color="#7d2830",
        fontsize=8,
    )


def _pearson_r(x: NDArray[np.float64], y: NDArray[np.float64]) -> float | None:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _finite_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(converted):
        return None
    return converted


def _sample_index(row: Mapping[str, object]) -> int | None:
    value = row.get("sample_index")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
