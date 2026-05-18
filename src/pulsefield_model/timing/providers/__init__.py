"""Timing providers."""

from pulsefield_model.timing.providers.beatthis import (
    BEATTHIS_FRAME_RATE_HZ,
    BEATTHIS_PROVIDER_NAME,
    DEFAULT_BEATTHIS_CHECKPOINT,
    DEFAULT_BEATTHIS_DEVICE,
    BeatThisAPI,
    BeatThisDependencyError,
    BeatThisTimingProvider,
)
from pulsefield_model.timing.providers.oracle import (
    DEFAULT_ORACLE_DENSE_TIMING_CACHE_CONFIG,
    DEFAULT_ORACLE_TIMING_CONFIG,
    ORACLE_TIMING_PROVIDER_NAME,
    OracleDenseTimingCacheConfig,
    OracleTimingConfig,
    OracleTimingProvider,
    fitted_timing_grid_from_red_points,
    load_or_create_oracle_dense_timing_v2_cache,
    oracle_timing_grid_from_beatmap,
    oracle_dense_timing_v2_cache_path,
    render_oracle_dense_timing_v2,
)

__all__ = [
    "BEATTHIS_FRAME_RATE_HZ",
    "BEATTHIS_PROVIDER_NAME",
    "DEFAULT_BEATTHIS_CHECKPOINT",
    "DEFAULT_BEATTHIS_DEVICE",
    "DEFAULT_ORACLE_DENSE_TIMING_CACHE_CONFIG",
    "DEFAULT_ORACLE_TIMING_CONFIG",
    "BeatThisAPI",
    "BeatThisDependencyError",
    "BeatThisTimingProvider",
    "ORACLE_TIMING_PROVIDER_NAME",
    "OracleDenseTimingCacheConfig",
    "OracleTimingConfig",
    "OracleTimingProvider",
    "fitted_timing_grid_from_red_points",
    "load_or_create_oracle_dense_timing_v2_cache",
    "oracle_timing_grid_from_beatmap",
    "oracle_dense_timing_v2_cache_path",
    "render_oracle_dense_timing_v2",
]
