"""Timing diagnostics."""

from pulsefield_model.timing.diagnostics.compare_to_oracle import (
    TimingGridComparison,
    compare_timing_grids,
    oracle_grid_from_red_timing_points,
    run_beatthis_oracle_comparison,
)
from pulsefield_model.timing.diagnostics.scatter import (
    DEFAULT_SCATTER_SPECS,
    ScatterPoints,
    ScatterSpec,
    load_report_json,
    write_scatter_artifacts,
)

__all__ = [
    "DEFAULT_SCATTER_SPECS",
    "ScatterPoints",
    "ScatterSpec",
    "TimingGridComparison",
    "compare_timing_grids",
    "load_report_json",
    "oracle_grid_from_red_timing_points",
    "run_beatthis_oracle_comparison",
    "write_scatter_artifacts",
]
