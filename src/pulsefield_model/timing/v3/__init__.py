"""Public Timing v3 representations, evidence extraction, and fitting APIs."""

from pulsefield_model.timing.v3.analytic_curve import (
    ConstantTempoSection,
    LinearTimeRampSection,
    PhaseContinuousTimingCurve,
    TimingCurveSeam,
    TimingCurveSection,
)
from pulsefield_model.timing.v3.audio_evidence import (
    RawAudioEvidence,
    RawAudioEvidenceConfig,
    RawAudioEvidenceRanking,
    extract_raw_audio_evidence,
)

from pulsefield_model.timing.v3.fitter import (
    TimingV3FitResult,
    TimingV3Fitter,
    TimingV3FitterConfig,
)
from pulsefield_model.timing.v3.inference import (
    DEFAULT_TIMING_MAX_SUPPORTED_AUDIO_DURATION_SECONDS,
    TIMING_INFERENCE_MODES,
    TIMING_MODE_V2,
    TIMING_MODE_V3_SHADOW,
    TimingEvidenceBundle,
    TimingInferenceMode,
    TimingV3Outcome,
    TimingV3Telemetry,
    run_timing_v3_shadow,
    unpack_packed_mel_20ms_to_log_mel_10ms,
)
from pulsefield_model.timing.v3.projection import (
    PROJECTION_METHOD_PRESERVE_ANCHORS,
    PROJECTION_METHOD_PRESERVE_BPM,
    TimingV3ProjectionResult,
    project_preserve_anchors,
    project_preserve_bpm,
)
from pulsefield_model.timing.v3.schema import ConstantTimingSection, TimingV3Grid
from pulsefield_model.timing.v3.tempo_track import (
    LocalTempoObservation,
    TempoTrackConfig,
    TempoTrackDiagnostics,
    TempoTrackProductionSelection,
    TempoTrackResult,
    TimingCandidateDiagnostic,
    generate_timing_candidates,
    tempo_track_result_to_dict,
)

__all__ = [
    "ConstantTempoSection",
    "ConstantTimingSection",
    "DEFAULT_TIMING_MAX_SUPPORTED_AUDIO_DURATION_SECONDS",
    "LinearTimeRampSection",
    "LocalTempoObservation",
    "PhaseContinuousTimingCurve",
    "PROJECTION_METHOD_PRESERVE_ANCHORS",
    "PROJECTION_METHOD_PRESERVE_BPM",
    "RawAudioEvidence",
    "RawAudioEvidenceConfig",
    "RawAudioEvidenceRanking",
    "TIMING_INFERENCE_MODES",
    "TIMING_MODE_V2",
    "TIMING_MODE_V3_SHADOW",
    "TempoTrackConfig",
    "TempoTrackDiagnostics",
    "TempoTrackProductionSelection",
    "TempoTrackResult",
    "TimingV3FitResult",
    "TimingV3Fitter",
    "TimingV3FitterConfig",
    "TimingV3Grid",
    "TimingV3ProjectionResult",
    "TimingCandidateDiagnostic",
    "TimingCurveSeam",
    "TimingCurveSection",
    "TimingEvidenceBundle",
    "TimingInferenceMode",
    "TimingV3Outcome",
    "TimingV3Telemetry",
    "extract_raw_audio_evidence",
    "generate_timing_candidates",
    "project_preserve_anchors",
    "project_preserve_bpm",
    "run_timing_v3_shadow",
    "tempo_track_result_to_dict",
    "unpack_packed_mel_20ms_to_log_mel_10ms",
]
