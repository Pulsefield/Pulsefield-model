from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


FloatArray: TypeAlias = NDArray[np.float32]
ComplexArray: TypeAlias = NDArray[np.complex64]
BoolArray: TypeAlias = NDArray[np.bool_]
TimeArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class MIRBackboneConfig:
    """Clock and analysis parameters for the deterministic MIR teacher.

    Mel and novelty features use a 5 ms clock.  The Fourier tempogram uses a
    20 ms output clock but multi-second centered analysis windows.  PLP is
    reconstructed on the 5 ms clock, one channel per analysis-window scale.
    The experiment default deliberately fixes one 8 s scale; shorter windows
    remain configurable only so numerical tests and later sensitivity runs do
    not need a second implementation.
    """

    sample_rate: int = 24_000
    mel_bins: int = 128
    mel_hop_ms: int = 5
    mel_window_ms: int = 40
    fmin_hz: float = 20.0
    fmax_hz: float = 12_000.0
    log_mel_floor: float = 1e-5
    novelty_band_edges_hz: tuple[float, ...] = (20.0, 200.0, 800.0, 3_200.0, 12_000.0)
    novelty_diff_frames: int = 1
    tempogram_hop_ms: int = 20
    novelty_local_average_ms: int = 200
    novelty_clip: float = 4.0
    tempogram_window_seconds: tuple[float, ...] = (8.0,)
    tempo_min_bpm: float = 30.0
    tempo_max_bpm: float = 600.0
    tempo_bins: int = 96

    def __post_init__(self) -> None:
        for name in ("sample_rate", "mel_bins", "mel_hop_ms", "mel_window_ms", "novelty_diff_frames"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        if not isinstance(self.tempogram_hop_ms, int) or isinstance(self.tempogram_hop_ms, bool):
            raise TypeError("tempogram_hop_ms must be an integer")
        if self.tempogram_hop_ms <= 0 or self.tempogram_hop_ms % self.mel_hop_ms != 0:
            raise ValueError("tempogram_hop_ms must be a positive multiple of mel_hop_ms")
        if self.sample_rate * self.mel_hop_ms % 1_000 != 0:
            raise ValueError("sample_rate and mel_hop_ms must produce an integer hop length")
        if self.sample_rate * self.mel_window_ms % 1_000 != 0:
            raise ValueError("sample_rate and mel_window_ms must produce an integer window length")
        if (
            isinstance(self.novelty_local_average_ms, bool)
            or not isinstance(self.novelty_local_average_ms, int)
            or self.novelty_local_average_ms < 0
            or self.novelty_local_average_ms % self.mel_hop_ms != 0
        ):
            raise ValueError("novelty_local_average_ms must be a non-negative multiple of mel_hop_ms")
        if not math.isfinite(self.novelty_clip) or self.novelty_clip <= 0.0:
            raise ValueError("novelty_clip must be positive and finite")
        if not (0.0 <= self.fmin_hz < self.fmax_hz <= self.sample_rate / 2):
            raise ValueError("Mel frequency range must lie inside [0, Nyquist]")
        if not math.isfinite(self.log_mel_floor) or self.log_mel_floor <= 0.0:
            raise ValueError("log_mel_floor must be positive and finite")
        if len(self.novelty_band_edges_hz) != 5:
            raise ValueError("novelty_band_edges_hz must contain five edges defining four bands")
        if any(not math.isfinite(edge) for edge in self.novelty_band_edges_hz):
            raise ValueError("novelty band edges must be finite")
        if any(left >= right for left, right in zip(self.novelty_band_edges_hz, self.novelty_band_edges_hz[1:])):
            raise ValueError("novelty band edges must be strictly increasing")
        if self.novelty_band_edges_hz[0] > self.fmin_hz or self.novelty_band_edges_hz[-1] < self.fmax_hz:
            raise ValueError("novelty bands must cover the configured Mel frequency range")
        if not self.tempogram_window_seconds:
            raise ValueError("tempogram_window_seconds must not be empty")
        if any(not math.isfinite(window) or window <= 0.0 for window in self.tempogram_window_seconds):
            raise ValueError("tempogram windows must be positive and finite")
        if (
            not math.isfinite(self.tempo_min_bpm)
            or not math.isfinite(self.tempo_max_bpm)
            or self.tempo_min_bpm <= 0.0
            or self.tempo_max_bpm <= self.tempo_min_bpm
        ):
            raise ValueError("tempo range must be positive, finite, and increasing")
        if not isinstance(self.tempo_bins, int) or isinstance(self.tempo_bins, bool) or self.tempo_bins < 2:
            raise ValueError("tempo_bins must be an integer of at least two")

    @property
    def mel_hop_seconds(self) -> float:
        return self.mel_hop_ms / 1_000.0

    @property
    def mel_window_seconds(self) -> float:
        return self.mel_window_ms / 1_000.0

    @property
    def mel_hop_length(self) -> int:
        return self.sample_rate * self.mel_hop_ms // 1_000

    @property
    def mel_window_length(self) -> int:
        return self.sample_rate * self.mel_window_ms // 1_000

    @property
    def tempogram_stride(self) -> int:
        return self.tempogram_hop_ms // self.mel_hop_ms

    @property
    def novelty_local_radius_frames(self) -> int:
        return self.novelty_local_average_ms // (2 * self.mel_hop_ms)

    @property
    def tempo_bpms(self) -> FloatArray:
        return np.geomspace(self.tempo_min_bpm, self.tempo_max_bpm, self.tempo_bins).astype(np.float32)


@dataclass(frozen=True)
class MIRBackbone:
    config: MIRBackboneConfig
    frame_centers_s: TimeArray
    log_mel: FloatArray
    mel_valid: BoolArray
    novelty: FloatArray
    novelty_valid: BoolArray
    tempogram_centers_s: TimeArray
    tempo_bpms: FloatArray
    tempogram: ComplexArray
    tempogram_valid: BoolArray
    signed_plp: FloatArray
    plp_valid: BoolArray


@dataclass(frozen=True)
class MIRProbeFeatures:
    """Fixed N/T/P feature groups consumed by the matched anchor probe."""

    fast_frame_centers_s: TimeArray
    slow_frame_centers_s: TimeArray
    acoustic: FloatArray
    novelty: FloatArray
    tempogram: FloatArray
    pulse: FloatArray
    acoustic_valid: BoolArray
    novelty_valid: BoolArray
    tempogram_valid: BoolArray
    pulse_valid: BoolArray


def compute_log_mel_5ms(
    waveform: object,
    *,
    sample_rate: int,
    config: MIRBackboneConfig = MIRBackboneConfig(),
) -> tuple[FloatArray, TimeArray, BoolArray]:
    """Compute centered log-Mel frames and their actual center times.

    The waveform is symmetrically positioned on a frame-center clock beginning
    at audio time zero. ``valid`` is false for frames whose 40 ms analysis
    window includes any left or right padding.
    """

    if sample_rate != config.sample_rate:
        raise ValueError(f"expected {config.sample_rate}Hz waveform, got {sample_rate}Hz")
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"waveform must be one-dimensional, got shape {audio.shape}")
    if not np.all(np.isfinite(audio)):
        raise ValueError("waveform must contain only finite values")
    if audio.size == 0:
        return (
            np.empty((0, config.mel_bins), dtype=np.float32),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.bool_),
        )

    hop = config.mel_hop_length
    window = config.mel_window_length
    frame_count = math.ceil(audio.size / hop)
    left_padding = window // 2
    required_samples = (frame_count - 1) * hop + window
    right_padding = max(0, required_samples - (audio.size + left_padding))
    padded = np.pad(audio, (left_padding, right_padding))

    import torch

    tensor = torch.from_numpy(padded).unsqueeze(0)
    with torch.no_grad():
        mel = _mel_layer(config)(tensor).squeeze(0).transpose(0, 1).cpu().numpy()
    log_mel = np.log(np.maximum(mel[:frame_count], config.log_mel_floor)).astype(np.float32, copy=False)
    centers_in_samples = np.arange(frame_count, dtype=np.float64) * hop
    starts = centers_in_samples - left_padding
    centers = centers_in_samples / sample_rate
    valid = ((starts >= 0) & (starts + window <= audio.size)).astype(np.bool_)
    return log_mel, centers, valid


def build_mir_backbone(
    waveform: object,
    *,
    sample_rate: int,
    config: MIRBackboneConfig = MIRBackboneConfig(),
) -> MIRBackbone:
    log_mel, centers, valid = compute_log_mel_5ms(
        waveform,
        sample_rate=sample_rate,
        config=config,
    )
    return build_mir_backbone_from_log_mel(
        log_mel,
        centers,
        frame_valid=valid,
        config=config,
    )


def spectral_flux_novelty(
    log_mel: object,
    *,
    frame_valid: object | None = None,
    config: MIRBackboneConfig = MIRBackboneConfig(),
) -> tuple[FloatArray, BoolArray]:
    """Return positive log-Mel flux as broadband plus four frequency bands."""

    mel = _as_log_mel(log_mel, config)
    valid = _as_frame_valid(frame_valid, mel.shape[0])
    raw_novelty = np.zeros((mel.shape[0], 5), dtype=np.float32)
    lag = config.novelty_diff_frames
    novelty_valid = np.zeros(mel.shape[0], dtype=np.bool_)
    if mel.shape[0] <= lag:
        return raw_novelty, novelty_valid

    positive_difference = np.maximum(mel[lag:] - mel[:-lag], 0.0)
    raw_novelty[lag:, 0] = positive_difference.mean(axis=1)
    frequencies = mel_center_frequencies_hz(config)
    for channel, (low, high) in enumerate(
        zip(config.novelty_band_edges_hz, config.novelty_band_edges_hz[1:]),
        start=1,
    ):
        if channel == 4:
            in_band = (frequencies >= low) & (frequencies <= high)
        else:
            in_band = (frequencies >= low) & (frequencies < high)
        if not np.any(in_band):
            raise ValueError(f"novelty band [{low}, {high}] contains no Mel bins")
        raw_novelty[lag:, channel] = positive_difference[:, in_band].mean(axis=1)
    novelty_valid[lag:] = valid[lag:] & valid[:-lag]

    radius = config.novelty_local_radius_frames
    if radius > 0:
        frame_count = raw_novelty.shape[0]
        indexes = np.arange(frame_count, dtype=np.int64)
        left = np.maximum(indexes - radius, 0)
        right = np.minimum(indexes + radius + 1, frame_count)
        prefix = np.vstack(
            (
                np.zeros((1, raw_novelty.shape[1]), dtype=np.float64),
                np.cumsum(raw_novelty, axis=0, dtype=np.float64),
            )
        )
        local_mean = (prefix[right] - prefix[left]) / (right - left)[:, None]
        novelty = np.maximum(raw_novelty - local_mean, 0.0).astype(np.float32)
        invalid_prefix = np.concatenate(([0], np.cumsum(~novelty_valid, dtype=np.int64)))
        novelty_valid &= (indexes >= radius) & (indexes + radius < frame_count)
        novelty_valid &= invalid_prefix[right] == invalid_prefix[left]
    else:
        novelty = raw_novelty

    for channel in range(novelty.shape[1]):
        positive = novelty[novelty_valid, channel]
        positive = positive[positive > 0.0]
        if positive.size:
            scale = float(np.quantile(positive, 0.99))
            if scale > np.finfo(np.float32).eps:
                novelty[:, channel] /= scale
    np.clip(novelty, 0.0, config.novelty_clip, out=novelty)
    novelty[~novelty_valid] = 0.0
    return novelty, novelty_valid


def complex_fourier_tempogram(
    novelty: object,
    frame_centers_s: object,
    *,
    frame_valid: object | None = None,
    config: MIRBackboneConfig = MIRBackboneConfig(),
) -> tuple[TimeArray, ComplexArray, BoolArray]:
    """Compute a centered Hann-window Fourier tempogram from broadband novelty.

    The first novelty channel is used when ``novelty`` has shape ``[time,
    channels]``.  Output has shape ``[time_20ms, scales, tempo_bins]``.  The
    validity mask has shape ``[time_20ms, scales]`` and requires the complete
    centered analysis window to contain valid novelty frames.
    """

    values = np.asarray(novelty, dtype=np.float32)
    if values.ndim == 2:
        if values.shape[1] < 1:
            raise ValueError("novelty must contain at least one channel")
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(f"novelty must have shape [time] or [time, channels], got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("novelty must contain only finite values")
    times = _as_uniform_frame_centers(frame_centers_s, values.shape[0], config.mel_hop_seconds)
    valid = _as_frame_valid(frame_valid, values.shape[0])
    output_times = times[:: config.tempogram_stride]
    scales = len(config.tempogram_window_seconds)
    bpms = config.tempo_bpms.astype(np.float64)
    coefficients = np.zeros((output_times.size, scales, bpms.size), dtype=np.complex64)
    output_valid = np.zeros((output_times.size, scales), dtype=np.bool_)
    if values.size == 0:
        return output_times, coefficients, output_valid

    invalid_prefix = np.concatenate(([0], np.cumsum(~valid, dtype=np.int64)))
    angular_frequencies = 2.0 * np.pi * bpms / 60.0
    signal = values.astype(np.float64)
    source_times = times.astype(np.float64)
    centers = output_times.astype(np.float64)

    for scale_index, duration in enumerate(config.tempogram_window_seconds):
        half_duration = duration / 2.0
        left, right = _centered_window_bounds(source_times, centers, half_duration)
        complete = (centers - half_duration >= source_times[0]) & (
            centers + half_duration <= source_times[-1]
        )
        complete &= invalid_prefix[right] == invalid_prefix[left]
        output_valid[:, scale_index] = complete
        hann_angular_frequency = 2.0 * np.pi / duration
        normalization = (
            0.5 * _rectangular_modulation_sum(np.ones_like(signal), source_times, centers, left, right, 0.0)
            + 0.25
            * _rectangular_modulation_sum(
                np.ones_like(signal), source_times, centers, left, right, -hann_angular_frequency
            )
            + 0.25
            * _rectangular_modulation_sum(
                np.ones_like(signal), source_times, centers, left, right, hann_angular_frequency
            )
        ).real
        normalization = np.maximum(normalization, np.finfo(np.float64).eps)
        for tempo_index, angular_frequency in enumerate(angular_frequencies):
            coefficient = (
                0.5
                * _rectangular_modulation_sum(
                    signal, source_times, centers, left, right, angular_frequency
                )
                + 0.25
                * _rectangular_modulation_sum(
                    signal,
                    source_times,
                    centers,
                    left,
                    right,
                    angular_frequency - hann_angular_frequency,
                )
                + 0.25
                * _rectangular_modulation_sum(
                    signal,
                    source_times,
                    centers,
                    left,
                    right,
                    angular_frequency + hann_angular_frequency,
                )
            )
            coefficient /= normalization
            coefficient[~complete] = 0.0
            coefficients[:, scale_index, tempo_index] = coefficient.astype(np.complex64)
    return output_times, coefficients, output_valid


def classical_plp(
    tempogram: object,
    tempogram_centers_s: object,
    target_frame_centers_s: object,
    *,
    tempogram_valid: object | None = None,
    target_frame_valid: object | None = None,
    config: MIRBackboneConfig = MIRBackboneConfig(),
) -> tuple[FloatArray, BoolArray]:
    """Return half-wave-rectified classical predominant local pulse."""

    signed, valid = _classical_plp_signed(
        tempogram,
        tempogram_centers_s,
        target_frame_centers_s,
        tempogram_valid=tempogram_valid,
        target_frame_valid=target_frame_valid,
        config=config,
    )
    return np.maximum(signed, 0.0).astype(np.float32, copy=False), valid


def _classical_plp_signed(
    tempogram: object,
    tempogram_centers_s: object,
    target_frame_centers_s: object,
    *,
    tempogram_valid: object | None = None,
    target_frame_valid: object | None = None,
    config: MIRBackboneConfig = MIRBackboneConfig(),
) -> tuple[FloatArray, BoolArray]:
    """Reconstruct signed predominant local pulse channels.

    At each tempogram frame and scale, the maximum-magnitude tempo coefficient
    defines one phase-aligned Hann-windowed sinusoid.  The kernels are
    overlap-added and normalized on the target clock.  Validity additionally
    requires the full two-window effective support around a target frame.
    """

    coefficients = np.asarray(tempogram, dtype=np.complex64)
    scale_count = len(config.tempogram_window_seconds)
    expected_tail = (scale_count, config.tempo_bins)
    if coefficients.ndim != 3 or coefficients.shape[1:] != expected_tail:
        raise ValueError(
            f"expected tempogram shape [time, {scale_count}, {config.tempo_bins}], "
            f"got {coefficients.shape}",
        )
    if not np.all(np.isfinite(coefficients.real)) or not np.all(np.isfinite(coefficients.imag)):
        raise ValueError("tempogram must contain only finite values")
    source_times = _as_strict_times(tempogram_centers_s, coefficients.shape[0], "tempogram_centers_s")
    target_times = _as_strict_times(target_frame_centers_s, None, "target_frame_centers_s")
    source_valid = _as_prefixed_valid(tempogram_valid, (coefficients.shape[0], scale_count))
    target_valid = _as_frame_valid(target_frame_valid, target_times.size)
    plp = np.zeros((target_times.size, scale_count), dtype=np.float64)
    weights = np.zeros_like(plp)
    validity = np.zeros((target_times.size, scale_count), dtype=np.bool_)
    bpms = config.tempo_bpms.astype(np.float64)

    for scale_index, duration in enumerate(config.tempogram_window_seconds):
        selected = np.argmax(np.abs(coefficients[:, scale_index]), axis=1)
        for source_index in np.flatnonzero(source_valid[:, scale_index]):
            center = source_times[source_index]
            half_duration = duration / 2.0
            left = int(np.searchsorted(target_times, center - half_duration, side="left"))
            right = int(np.searchsorted(target_times, center + half_duration, side="right"))
            if left == right:
                continue
            relative_times = target_times[left:right] - center
            window = 0.5 + 0.5 * np.cos(2.0 * np.pi * relative_times / duration)
            tempo_index = selected[source_index]
            omega = 2.0 * np.pi * bpms[tempo_index] / 60.0
            local_pulse = np.real(
                coefficients[source_index, scale_index, tempo_index]
                * np.exp(1j * omega * relative_times)
            )
            plp[left:right, scale_index] += window * local_pulse
            weights[left:right, scale_index] += window

        has_weight = weights[:, scale_index] > np.finfo(np.float64).eps
        plp[has_weight, scale_index] /= weights[has_weight, scale_index]
        if target_times.size:
            full_support = (target_times - duration >= target_times[0]) & (
                target_times + duration <= target_times[-1]
            )
            target_invalid_prefix = np.concatenate(([0], np.cumsum(~target_valid, dtype=np.int64)))
            left, right = _centered_window_bounds(target_times, target_times, duration)
            full_support &= target_invalid_prefix[right] == target_invalid_prefix[left]
            validity[:, scale_index] = has_weight & full_support
        plp[~validity[:, scale_index], scale_index] = 0.0
    return plp.astype(np.float32), validity


def interpolate_frame_centers(
    values: object,
    source_frame_centers_s: object,
    target_times_s: object,
    *,
    valid: object | None = None,
) -> tuple[np.ndarray, BoolArray]:
    """Linearly interpolate features without extrapolating across invalid frames.

    A target exactly on a source center needs only that source frame to be
    valid.  Other targets require both bracketing source frames to be valid.
    The validity array may be ``[time]`` or a prefix of the feature shape, for
    example ``[time, scale]`` for ``[time, scale, tempo]`` coefficients.
    """

    array = np.asarray(values)
    if array.ndim < 1:
        raise ValueError("values must have a time dimension")
    if not (
        np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise TypeError("values must be floating-point or complex")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must contain only finite values")
    source = _as_strict_times(source_frame_centers_s, array.shape[0], "source_frame_centers_s")
    targets = np.asarray(target_times_s, dtype=np.float64)
    if targets.ndim != 1 or not np.all(np.isfinite(targets)):
        raise ValueError("target_times_s must be a finite one-dimensional array")
    validity = _as_interpolation_valid(valid, array.shape)
    output_shape = (targets.size, *array.shape[1:])
    output = np.zeros(output_shape, dtype=array.dtype)
    output_valid = np.zeros((targets.size, *validity.shape[1:]), dtype=np.bool_)
    if source.size == 0 or targets.size == 0:
        return output, output_valid

    right = np.searchsorted(source, targets, side="left")
    exact = np.zeros(targets.size, dtype=np.bool_)
    in_right = right < source.size
    exact[in_right] = np.isclose(source[right[in_right]], targets[in_right], rtol=0.0, atol=1e-9)
    exact_indexes = np.flatnonzero(exact)
    if exact_indexes.size:
        source_indexes = right[exact_indexes]
        output[exact_indexes] = array[source_indexes]
        output_valid[exact_indexes] = validity[source_indexes]

    between = (~exact) & (right > 0) & (right < source.size)
    target_indexes = np.flatnonzero(between)
    if target_indexes.size:
        upper = right[target_indexes]
        lower = upper - 1
        alpha = (targets[target_indexes] - source[lower]) / (source[upper] - source[lower])
        alpha_shape = (alpha.size, *([1] * (array.ndim - 1)))
        output[target_indexes] = (
            array[lower] * (1.0 - alpha.reshape(alpha_shape))
            + array[upper] * alpha.reshape(alpha_shape)
        )
        output_valid[target_indexes] = validity[lower] & validity[upper]

    expanded_valid = output_valid
    while expanded_valid.ndim < output.ndim:
        expanded_valid = np.expand_dims(expanded_valid, axis=-1)
    output = np.where(expanded_valid, output, np.zeros((), dtype=output.dtype))
    return output, output_valid


def build_mir_backbone_from_log_mel(
    log_mel: object,
    frame_centers_s: object,
    *,
    frame_valid: object | None = None,
    config: MIRBackboneConfig = MIRBackboneConfig(),
) -> MIRBackbone:
    """Build the deterministic teacher from precomputed 5 ms log-Mel."""

    mel = _as_log_mel(log_mel, config)
    centers = _as_uniform_frame_centers(frame_centers_s, mel.shape[0], config.mel_hop_seconds)
    mel_valid = _as_frame_valid(frame_valid, mel.shape[0])
    novelty, novelty_valid = spectral_flux_novelty(mel, frame_valid=mel_valid, config=config)
    tempogram_centers, tempogram, tempogram_valid = complex_fourier_tempogram(
        novelty,
        centers,
        frame_valid=novelty_valid,
        config=config,
    )
    signed_plp, plp_valid = _classical_plp_signed(
        tempogram,
        tempogram_centers,
        centers,
        tempogram_valid=tempogram_valid,
        target_frame_valid=novelty_valid,
        config=config,
    )
    return MIRBackbone(
        config=config,
        frame_centers_s=centers,
        log_mel=mel,
        mel_valid=mel_valid,
        novelty=novelty,
        novelty_valid=novelty_valid,
        tempogram_centers_s=tempogram_centers,
        tempo_bpms=config.tempo_bpms,
        tempogram=tempogram,
        tempogram_valid=tempogram_valid,
        signed_plp=signed_plp,
        plp_valid=plp_valid,
    )


def mir_probe_features(
    backbone: MIRBackbone,
    *,
    config: MIRBackboneConfig | None = None,
) -> MIRProbeFeatures:
    """Project the deterministic teacher into the card's fixed A/N/T/P groups.

    The probe uses exactly one tempogram support scale. Periodicity features
    contain 96 absolute-tempo magnitudes, 24 octave-folded bins, log total
    magnitude, and normalized entropy. Pulse features contain signed and
    rectified PLP plus confidence, normalized log tempo, and sine/cosine phase.
    """

    if config is None:
        config = backbone.config
    elif config != backbone.config:
        raise ValueError("probe config must exactly match the backbone extraction config")
    if len(config.tempogram_window_seconds) != 1:
        raise ValueError("probe features require exactly one tempogram support scale")
    expected_tempogram_shape = (
        backbone.tempogram_centers_s.size,
        1,
        config.tempo_bins,
    )
    if backbone.tempogram.shape != expected_tempogram_shape:
        raise ValueError(
            f"backbone tempogram shape {backbone.tempogram.shape} does not match config {expected_tempogram_shape}",
        )

    acoustic = np.array(backbone.log_mel, dtype=np.float32, copy=True)
    acoustic[~backbone.mel_valid] = 0.0
    novelty = np.array(backbone.novelty, dtype=np.float32, copy=True)
    novelty[~backbone.novelty_valid] = 0.0

    magnitudes = np.abs(backbone.tempogram[:, 0]).astype(np.float32)
    total = magnitudes.sum(axis=1, keepdims=True)
    posterior = np.divide(
        magnitudes,
        total,
        out=np.zeros_like(magnitudes),
        where=total > np.finfo(np.float32).eps,
    )
    entropy = -np.sum(posterior * np.log(np.maximum(posterior, np.finfo(np.float32).tiny)), axis=1)
    entropy /= math.log(config.tempo_bins)
    cyclic = np.zeros((magnitudes.shape[0], 24), dtype=np.float32)
    octave_positions = np.mod(np.log2(config.tempo_bpms / config.tempo_min_bpm), 1.0)
    cyclic_indexes = np.minimum((octave_positions * 24).astype(np.int64), 23)
    for tempo_index, cyclic_index in enumerate(cyclic_indexes):
        cyclic[:, cyclic_index] += posterior[:, tempo_index]
    tempogram_features = np.concatenate(
        (
            np.log1p(magnitudes),
            cyclic,
            np.log1p(total),
            entropy[:, None].astype(np.float32),
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    tempogram_valid = np.asarray(backbone.tempogram_valid[:, 0], dtype=np.bool_)
    tempogram_features[~tempogram_valid] = 0.0

    signed_plp = np.asarray(backbone.signed_plp, dtype=np.float32)
    signed_valid = np.asarray(backbone.plp_valid, dtype=np.bool_)
    if signed_plp.shape != (backbone.frame_centers_s.size, 1) or signed_valid.shape != signed_plp.shape:
        raise ValueError("signed PLP values and validity must have shape [fast_time, 1]")
    selected_indexes = np.argmax(magnitudes, axis=1)
    selected = np.take_along_axis(backbone.tempogram[:, 0], selected_indexes[:, None], axis=1)[:, 0]
    selected_magnitude = np.abs(selected)
    confidence = np.divide(
        selected_magnitude,
        total[:, 0],
        out=np.zeros_like(selected_magnitude, dtype=np.float32),
        where=total[:, 0] > np.finfo(np.float32).eps,
    )
    log_tempo = np.log(config.tempo_bpms[selected_indexes])
    log_tempo = (log_tempo - math.log(config.tempo_min_bpm)) / math.log(
        config.tempo_max_bpm / config.tempo_min_bpm,
    )
    unit_phase = np.divide(
        selected,
        selected_magnitude,
        out=np.zeros_like(selected),
        where=selected_magnitude > np.finfo(np.float32).eps,
    )
    slow_pulse = np.column_stack(
        (confidence, log_tempo, unit_phase.imag, unit_phase.real),
    ).astype(np.float32)
    interpolated_pulse, interpolated_valid = interpolate_frame_centers(
        slow_pulse,
        backbone.tempogram_centers_s,
        backbone.frame_centers_s,
        valid=tempogram_valid,
    )
    pulse_valid = signed_valid[:, 0] & interpolated_valid
    pulse = np.column_stack(
        (
            signed_plp[:, 0],
            np.maximum(signed_plp[:, 0], 0.0),
            interpolated_pulse,
        ),
    ).astype(np.float32)
    pulse[~pulse_valid] = 0.0

    return MIRProbeFeatures(
        fast_frame_centers_s=backbone.frame_centers_s,
        slow_frame_centers_s=backbone.tempogram_centers_s,
        acoustic=acoustic,
        novelty=novelty,
        tempogram=tempogram_features,
        pulse=pulse,
        acoustic_valid=np.asarray(backbone.mel_valid, dtype=np.bool_),
        novelty_valid=np.asarray(backbone.novelty_valid, dtype=np.bool_),
        tempogram_valid=tempogram_valid,
        pulse_valid=pulse_valid,
    )


def mel_center_frequencies_hz(config: MIRBackboneConfig = MIRBackboneConfig()) -> FloatArray:
    """Return Slaney-Mel filter centers matching nnAudio's ``htk=False`` mode."""

    minimum = _hz_to_slaney_mel(config.fmin_hz)
    maximum = _hz_to_slaney_mel(config.fmax_hz)
    points = np.linspace(minimum, maximum, config.mel_bins + 2, dtype=np.float64)[1:-1]
    return _slaney_mel_to_hz(points).astype(np.float32)


@lru_cache(maxsize=4)
def _mel_layer(config: MIRBackboneConfig):
    from nnAudio.features import MelSpectrogram

    return MelSpectrogram(
        sr=config.sample_rate,
        n_fft=config.mel_window_length,
        win_length=config.mel_window_length,
        n_mels=config.mel_bins,
        hop_length=config.mel_hop_length,
        window="hann",
        center=False,
        power=2.0,
        htk=False,
        fmin=config.fmin_hz,
        fmax=config.fmax_hz,
        norm=1,
        trainable_mel=False,
        trainable_STFT=False,
        verbose=False,
    )


def _rectangular_modulation_sum(
    signal: NDArray[np.float64],
    source_times: NDArray[np.float64],
    centers: NDArray[np.float64],
    left: NDArray[np.int64],
    right: NDArray[np.int64],
    angular_frequency: float,
) -> NDArray[np.complex128]:
    modulated = signal * np.exp(-1j * angular_frequency * source_times)
    prefix = np.empty(modulated.size + 1, dtype=np.complex128)
    prefix[0] = 0.0
    np.cumsum(modulated, out=prefix[1:])
    return np.exp(1j * angular_frequency * centers) * (prefix[right] - prefix[left])


def _centered_window_bounds(
    source_times: NDArray[np.float64],
    centers: NDArray[np.float64],
    half_width: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    left = np.searchsorted(source_times, centers - half_width, side="left")
    right = np.searchsorted(source_times, centers + half_width, side="right")
    return left.astype(np.int64), right.astype(np.int64)


def _as_log_mel(log_mel: object, config: MIRBackboneConfig) -> FloatArray:
    mel = np.asarray(log_mel, dtype=np.float32)
    if mel.ndim != 2 or mel.shape[1] != config.mel_bins:
        raise ValueError(f"expected log-Mel shape [time, {config.mel_bins}], got {mel.shape}")
    if not np.all(np.isfinite(mel)):
        raise ValueError("log-Mel must contain only finite values")
    return mel


def _as_frame_valid(valid: object | None, frame_count: int) -> BoolArray:
    if valid is None:
        return np.ones(frame_count, dtype=np.bool_)
    array = np.asarray(valid, dtype=np.bool_)
    if array.shape != (frame_count,):
        raise ValueError(f"expected validity shape {(frame_count,)}, got {array.shape}")
    return array


def _as_prefixed_valid(valid: object | None, expected_shape: tuple[int, ...]) -> BoolArray:
    if valid is None:
        return np.ones(expected_shape, dtype=np.bool_)
    array = np.asarray(valid, dtype=np.bool_)
    if array.shape != expected_shape:
        raise ValueError(f"expected validity shape {expected_shape}, got {array.shape}")
    return array


def _as_interpolation_valid(valid: object | None, value_shape: tuple[int, ...]) -> BoolArray:
    if valid is None:
        return np.ones(value_shape[0], dtype=np.bool_)
    array = np.asarray(valid, dtype=np.bool_)
    if array.ndim < 1 or array.shape[0] != value_shape[0] or array.ndim > len(value_shape):
        raise ValueError("valid must share the values time dimension and be a feature-shape prefix")
    if array.shape[1:] != value_shape[1 : array.ndim]:
        raise ValueError("valid trailing dimensions must be a prefix of the values feature dimensions")
    return array


def _as_strict_times(times: object, expected_count: int | None, name: str) -> NDArray[np.float64]:
    array = np.asarray(times, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    if expected_count is not None and array.shape != (expected_count,):
        raise ValueError(f"expected {name} shape {(expected_count,)}, got {array.shape}")
    if array.size > 1 and np.any(np.diff(array) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return array


def _as_uniform_frame_centers(
    times: object,
    expected_count: int,
    expected_hop_s: float,
) -> NDArray[np.float64]:
    array = _as_strict_times(times, expected_count, "frame_centers_s")
    tolerance = max(1e-7, expected_hop_s * 0.01)
    if array.size > 1 and not np.allclose(np.diff(array), expected_hop_s, rtol=0.0, atol=tolerance):
        raise ValueError(f"frame_centers_s must use the configured {expected_hop_s:g}s hop")
    return array


def _hz_to_slaney_mel(frequency_hz: float) -> float:
    linear_hz_per_mel = 200.0 / 3.0
    minimum_log_hz = 1_000.0
    minimum_log_mel = minimum_log_hz / linear_hz_per_mel
    log_step = math.log(6.4) / 27.0
    if frequency_hz < minimum_log_hz:
        return frequency_hz / linear_hz_per_mel
    return minimum_log_mel + math.log(frequency_hz / minimum_log_hz) / log_step


def _slaney_mel_to_hz(mels: NDArray[np.float64]) -> NDArray[np.float64]:
    linear_hz_per_mel = 200.0 / 3.0
    minimum_log_hz = 1_000.0
    minimum_log_mel = minimum_log_hz / linear_hz_per_mel
    log_step = math.log(6.4) / 27.0
    frequencies = linear_hz_per_mel * mels
    logarithmic = mels >= minimum_log_mel
    frequencies[logarithmic] = minimum_log_hz * np.exp(
        log_step * (mels[logarithmic] - minimum_log_mel)
    )
    return frequencies
