from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint


ACOUSTIC_GROUP = "A"
MIR_GROUPS = frozenset(("N", "T", "P"))
ALL_FEATURE_GROUPS = frozenset((ACOUSTIC_GROUP, *MIR_GROUPS))
FEATURE_GROUP_PAIRS = tuple(combinations(sorted(ALL_FEATURE_GROUPS), 2))


@dataclass(frozen=True)
class MirAnchorProbeConfig:
    acoustic_dim: int = 128
    novelty_dim: int = 5
    tempogram_dim: int = 122
    pulse_dim: int = 6
    history_dim: int = 10
    candidate_dim: int = 6
    history_hidden: int = 64
    encoder_width: int = 48
    embedding_dim: int = 32
    interaction_rank: int = 16
    acoustic_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    high_rate_dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    tempogram_dilations: tuple[int, ...] = (1, 2, 4)
    acoustic_kernel: int = 3
    acoustic_stride: int = 4
    acoustic_pool_kernel: int = 5
    high_rate_kernel: int = 5
    tempogram_kernel: int = 3
    dropout: float = 0.1
    max_parameters: int = 300_000

    @property
    def acoustic_radius_frames(self) -> int:
        return (self.acoustic_kernel - 1) * sum(self.acoustic_dilations) // 2

    @property
    def high_rate_radius_frames(self) -> int:
        return (self.high_rate_kernel - 1) * sum(self.high_rate_dilations) // 2

    @property
    def tempogram_radius_frames(self) -> int:
        return (self.tempogram_kernel - 1) * sum(self.tempogram_dilations) // 2


class MirAnchorProbe(nn.Module):
    """Research-only group-structured scorer for audio feature sequences.

    Audio sequences are encoded once per song with :meth:`encode_group`.
    Callers interpolate the encoded sequences at candidate support points, then
    pass those gathered embeddings to :meth:`forward`. This keeps continuous
    timestamp lookup outside the model and avoids re-encoding local windows for
    every matched alternative. Unary, history-conditioned, and pairwise audio
    terms support coalition attribution without an unrestricted note decoder.
    """

    def __init__(self, config: MirAnchorProbeConfig = MirAnchorProbeConfig()) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.history_encoder = nn.GRU(
            input_size=config.history_dim,
            hidden_size=config.history_hidden,
            batch_first=True,
        )
        group_dims = {
            "A": config.acoustic_dim,
            "N": config.novelty_dim,
            "T": config.tempogram_dim,
            "P": config.pulse_dim,
        }
        group_dilations = {
            "A": config.acoustic_dilations,
            "N": config.high_rate_dilations,
            "T": config.tempogram_dilations,
            "P": config.high_rate_dilations,
        }
        group_kernels = {
            "A": config.acoustic_kernel,
            "N": config.high_rate_kernel,
            "T": config.tempogram_kernel,
            "P": config.high_rate_kernel,
        }
        group_strides = {"A": config.acoustic_stride, "N": 1, "T": 1, "P": 1}
        group_pool_kernels = {"A": config.acoustic_pool_kernel, "N": 1, "T": 1, "P": 1}
        self.feature_encoders = nn.ModuleDict(
            {
                group: _TemporalFeatureEncoder(
                    input_dim=input_dim,
                    width=config.encoder_width,
                    output_dim=config.embedding_dim,
                    kernel_size=group_kernels[group],
                    dilations=group_dilations[group],
                    dropout=config.dropout,
                    input_stride=group_strides[group],
                    input_pool_kernel=group_pool_kernels[group],
                )
                for group, input_dim in group_dims.items()
            },
        )
        base_input_dim = config.history_hidden + config.candidate_dim
        self.base_head = nn.Sequential(
            nn.Linear(base_input_dim, 64),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, 1),
        )
        self.unary_heads = nn.ModuleDict(
            {
                group: nn.Sequential(
                    nn.Linear(config.embedding_dim, 32),
                    nn.SiLU(),
                    nn.Linear(32, 1),
                )
                for group in sorted(ALL_FEATURE_GROUPS)
            },
        )
        self.history_interactions = nn.ModuleDict(
            {
                group: nn.Linear(config.history_hidden, config.interaction_rank, bias=False)
                for group in sorted(ALL_FEATURE_GROUPS)
            },
        )
        self.audio_interactions = nn.ModuleDict(
            {
                group: nn.Linear(config.embedding_dim, config.interaction_rank, bias=False)
                for group in sorted(ALL_FEATURE_GROUPS)
            },
        )
        self.interaction_outputs = nn.ModuleDict(
            {
                group: nn.Linear(config.interaction_rank, 1, bias=False)
                for group in sorted(ALL_FEATURE_GROUPS)
            },
        )
        self.feature_pair_heads = nn.ModuleDict(
            {
                f"{left}:{right}": nn.Bilinear(
                    config.embedding_dim,
                    config.embedding_dim,
                    1,
                    bias=False,
                )
                for left, right in FEATURE_GROUP_PAIRS
            },
        )
        if self.parameter_count() > config.max_parameters:
            raise ValueError(
                f"probe has {self.parameter_count()} parameters, exceeding max_parameters={config.max_parameters}",
            )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def encode_group(
        self,
        group: str,
        features: torch.Tensor,
        *,
        prepooled: bool = False,
    ) -> torch.Tensor:
        if group not in ALL_FEATURE_GROUPS:
            raise ValueError(f"unknown feature group {group!r}; expected one of {sorted(ALL_FEATURE_GROUPS)}")
        return self.feature_encoders[group](features, prepooled=prepooled)

    def encode_group_chunked(
        self,
        group: str,
        features: torch.Tensor,
        *,
        lengths: Sequence[int] | torch.Tensor,
        chunk_size: int,
        bank_capacity: int,
        prepooled: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch into fixed-capacity storage using fixed-size cores.

        ``features`` has shape ``[B,T,input_dim]`` and ``lengths`` gives the
        valid prefix of each batch row. The returned bank is flattened to
        ``[B * bank_capacity, embedding_dim]``; the returned ``[B]`` lengths
        identify each row's valid prefix when the bank is reshaped. Storage
        after every valid prefix is zero.

        Strided encoders require caller-side pooling and ``prepooled=True``.
        This keeps every learned operation on a fixed-size core. In evaluation
        mode, or when dropout is zero, valid outputs and gradients are
        equivalent to encoding each unpadded sequence with :meth:`encode_group`.
        """

        if group not in ALL_FEATURE_GROUPS:
            raise ValueError(f"unknown feature group {group!r}; expected one of {sorted(ALL_FEATURE_GROUPS)}")
        return self.feature_encoders[group].forward_chunked(
            features,
            lengths=lengths,
            chunk_size=chunk_size,
            bank_capacity=bank_capacity,
            prepooled=prepooled,
        )

    def encode_history(self, history: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Encode valid, oldest-to-newest rows followed by padding.

        Empty histories are supported and return an all-zero state. Padded rows
        are computed only to keep the GRU input shape fixed; the state is read
        immediately after the last valid row, so later padding cannot affect it.
        """

        if history.ndim != 3 or history.shape[-1] != self.config.history_dim:
            raise ValueError(
                f"history must have shape [B,K,{self.config.history_dim}], got {tuple(history.shape)}",
            )
        if padding_mask.shape != history.shape[:2] or padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be bool with shape [B,K]")
        if torch.any(padding_mask[:, :-1] & ~padding_mask[:, 1:]):
            raise ValueError("history padding must follow all valid rows")

        lengths = (~padding_mask).sum(dim=1)
        outputs, _ = self.history_encoder(history)
        gather_indexes = lengths.sub(1).clamp_min(0)
        gather_indexes = gather_indexes[:, None, None].expand(-1, 1, self.config.history_hidden)
        state = outputs.gather(1, gather_indexes).squeeze(1)
        return state.masked_fill((lengths == 0).unsqueeze(1), 0.0)

    def forward(
        self,
        *,
        history_state: torch.Tensor,
        candidate_features: torch.Tensor,
        embeddings: Mapping[str, torch.Tensor],
        coalition: Sequence[str] = (),
    ) -> torch.Tensor:
        """Score candidate support points.

        Shapes use arbitrary shared leading dimensions. ``history_state`` ends
        in ``history_hidden`` and may omit the final support dimension; it is
        broadcast across that dimension. Candidate features contain elapsed
        time and song-relative position. Every embedding ends in
        ``embedding_dim``.
        """

        selected = frozenset(coalition)
        unknown = selected - ALL_FEATURE_GROUPS
        if unknown:
            raise ValueError(f"unknown audio groups: {sorted(unknown)}")
        missing = selected - embeddings.keys()
        if missing:
            raise ValueError(f"missing embeddings for groups: {sorted(missing)}")
        if candidate_features.shape[-1] != self.config.candidate_dim:
            raise ValueError(f"candidate_features must end in {self.config.candidate_dim} values")
        if history_state.shape[-1] != self.config.history_hidden:
            raise ValueError(f"history_state must end in {self.config.history_hidden} values")

        history = _broadcast_history(history_state, candidate_features)
        score = self.base_head(torch.cat((history, candidate_features), dim=-1)).squeeze(-1)
        for group in sorted(selected):
            embedding = embeddings[group]
            _validate_embedding(embedding, self.config.embedding_dim, group=group)
            expected_shape = (*candidate_features.shape[:-1], self.config.embedding_dim)
            if embedding.shape != expected_shape:
                raise ValueError(f"embedding group {group!r} must have shape {expected_shape}")
            unary = self.unary_heads[group](embedding).squeeze(-1)
            interaction = self.interaction_outputs[group](
                self.history_interactions[group](history) * self.audio_interactions[group](embedding),
            ).squeeze(-1)
            score = score + unary + interaction
        for left, right in FEATURE_GROUP_PAIRS:
            if left in selected and right in selected:
                score = score + self.feature_pair_heads[f"{left}:{right}"](
                    embeddings[left],
                    embeddings[right],
                ).squeeze(-1)
        return score


class _TemporalFeatureEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        width: int,
        output_dim: int,
        kernel_size: int,
        dilations: Sequence[int],
        dropout: float,
        input_stride: int,
        input_pool_kernel: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.input_stride = input_stride
        self.input_pool_kernel = input_pool_kernel
        self.input_projection = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(
            [
                _DepthwiseResidualBlock(
                    width=width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ],
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, output_dim)

    def forward(self, features: torch.Tensor, *, prepooled: bool = False) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(f"features must have shape [B,T,{self.input_dim}], got {tuple(features.shape)}")
        if not isinstance(prepooled, bool):
            raise ValueError("prepooled must be bool")
        if self.input_stride > 1 and not prepooled:
            features = F.avg_pool1d(
                features.transpose(1, 2),
                kernel_size=self.input_pool_kernel,
                stride=self.input_stride,
                padding=self.input_pool_kernel // 2,
                count_include_pad=False,
            ).transpose(1, 2)
        hidden = self.input_projection(features)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(self.output_norm(hidden))

    def forward_chunked(
        self,
        features: torch.Tensor,
        *,
        lengths: Sequence[int] | torch.Tensor,
        chunk_size: int,
        bank_capacity: int,
        prepooled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(f"features must have shape [B,T,{self.input_dim}], got {tuple(features.shape)}")
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("features must contain at least one batch row and time frame")
        if not isinstance(prepooled, bool):
            raise ValueError("prepooled must be bool")
        if self.input_stride > 1 and not prepooled:
            raise ValueError("chunked encoding for a strided input requires prepooled=True")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        max_block_radius = max(
            block.depthwise.dilation[0] * (block.depthwise.kernel_size[0] // 2)
            for block in self.blocks
        )
        if chunk_size <= max_block_radius:
            raise ValueError(
                f"chunk_size must exceed the largest block radius ({max_block_radius}), got {chunk_size}",
            )
        if isinstance(bank_capacity, bool) or not isinstance(bank_capacity, int) or bank_capacity <= 0:
            raise ValueError("bank_capacity must be a positive integer")
        if features.shape[1] > bank_capacity:
            raise ValueError(
                f"feature time dimension {features.shape[1]} exceeds bank_capacity={bank_capacity}",
            )

        encoded_lengths = _validate_chunk_lengths(
            lengths,
            batch_size=features.shape[0],
            feature_frames=features.shape[1],
            bank_capacity=bank_capacity,
            device=features.device,
        )
        chunk_count = math.ceil(bank_capacity / chunk_size)
        working_capacity = chunk_count * chunk_size
        valid = torch.arange(working_capacity, device=features.device).unsqueeze(0)
        valid = valid < encoded_lengths.unsqueeze(1)

        hidden_cores = []
        for start in range(0, working_capacity, chunk_size):
            source_frames = min(chunk_size, max(0, features.shape[1] - start))
            if source_frames == chunk_size:
                input_core = features[:, start : start + chunk_size].clone()
            else:
                input_core = features.new_zeros((features.shape[0], chunk_size, self.input_dim))
                if source_frames > 0:
                    input_core[:, :source_frames] = features[:, start : start + source_frames]
                input_core = input_core.clone()
            core_valid = valid[:, start : start + chunk_size, None]
            hidden = self.input_projection(input_core)
            hidden_cores.append(hidden.masked_fill(~core_valid, 0.0))

        def apply_blocks(*cores: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return self._forward_chunked_blocks(
                cores,
                valid=valid,
                chunk_size=chunk_size,
            )

        if self.training and torch.is_grad_enabled():
            hidden_cores = list(
                _checkpoint(
                    apply_blocks,
                    *hidden_cores,
                    use_reentrant=False,
                    preserve_rng_state=True,
                ),
            )
        else:
            hidden_cores = list(apply_blocks(*hidden_cores))

        output_chunks = []
        for index, hidden in enumerate(hidden_cores):
            core = self.output_projection(self.output_norm(hidden))
            core_valid = valid[:, index * chunk_size : (index + 1) * chunk_size, None]
            output_chunks.append(core.masked_fill(~core_valid, 0.0))
        output = torch.cat(output_chunks, dim=1)[:, :bank_capacity]
        return output.reshape(features.shape[0] * bank_capacity, -1), encoded_lengths

    def _forward_chunked_blocks(
        self,
        hidden_cores: Sequence[torch.Tensor],
        *,
        valid: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, ...]:
        chunk_count = len(hidden_cores)
        for block in self.blocks:
            normalized_cores = []
            for index, hidden in enumerate(hidden_cores):
                core_valid = valid[:, index * chunk_size : (index + 1) * chunk_size, None]
                normalized_cores.append(block.norm(hidden).masked_fill(~core_valid, 0.0))

            radius = block.depthwise.dilation[0] * (block.depthwise.kernel_size[0] // 2)
            zero_halo = hidden_cores[0].new_zeros(
                (hidden_cores[0].shape[0], radius, hidden_cores[0].shape[-1]),
            )
            output_cores = []
            for index, normalized in enumerate(normalized_cores):
                if radius == 0:
                    halo = normalized
                else:
                    left = normalized_cores[index - 1][:, -radius:] if index > 0 else zero_halo
                    right = normalized_cores[index + 1][:, :radius] if index + 1 < chunk_count else zero_halo
                    halo = torch.cat((left, normalized, right), dim=1)
                core = F.conv1d(
                    halo.transpose(1, 2),
                    block.depthwise.weight,
                    block.depthwise.bias,
                    padding=0,
                    dilation=block.depthwise.dilation,
                    groups=block.depthwise.groups,
                ).transpose(1, 2)
                core = block.expand(core)
                core = block.contract(block.dropout(block.activation(core)))
                core = hidden_cores[index] + block.dropout(core)
                core_valid = valid[:, index * chunk_size : (index + 1) * chunk_size, None]
                output_cores.append(core.masked_fill(~core_valid, 0.0))
            hidden_cores = output_cores
        return tuple(hidden_cores)


class _DepthwiseResidualBlock(nn.Module):
    def __init__(self, *, width: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size // 2),
            groups=width,
        )
        self.expand = nn.Linear(width, 2 * width)
        self.contract = nn.Linear(2 * width, width)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.depthwise(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.expand(hidden)
        hidden = self.contract(self.dropout(self.activation(hidden)))
        return residual + self.dropout(hidden)


def _validate_chunk_lengths(
    lengths: Sequence[int] | torch.Tensor,
    *,
    batch_size: int,
    feature_frames: int,
    bank_capacity: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(lengths, torch.Tensor):
        if lengths.ndim != 1:
            raise ValueError("lengths must be a one-dimensional integer tensor or sequence")
        if lengths.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("lengths must contain integers")
        values = lengths.detach().cpu().tolist()
    else:
        try:
            values = list(lengths)
        except TypeError as exc:
            raise ValueError("lengths must be a one-dimensional integer tensor or sequence") from exc
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("lengths must contain integers")
    if len(values) != batch_size:
        raise ValueError(f"lengths must contain one value per batch row ({batch_size}), got {len(values)}")
    if any(value <= 0 for value in values):
        raise ValueError("lengths must be positive")
    if any(value > feature_frames for value in values):
        raise ValueError(f"lengths cannot exceed the feature time dimension ({feature_frames})")
    if any(value > bank_capacity for value in values):
        raise ValueError(f"lengths cannot exceed bank_capacity={bank_capacity}")
    return torch.tensor(values, dtype=torch.long, device=device)


def _broadcast_history(history: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    if history.ndim == embedding.ndim - 1:
        history = history.unsqueeze(-2)
    try:
        return torch.broadcast_to(history, (*embedding.shape[:-1], history.shape[-1]))
    except RuntimeError as exc:
        raise ValueError("history_state cannot broadcast to candidate embedding dimensions") from exc


def _validate_embedding(embedding: torch.Tensor, embedding_dim: int, *, group: str) -> None:
    if embedding.shape[-1] != embedding_dim:
        raise ValueError(f"embedding group {group!r} must end in {embedding_dim} values")


def _validate_config(config: MirAnchorProbeConfig) -> None:
    positive_ints = {
        "acoustic_dim": config.acoustic_dim,
        "novelty_dim": config.novelty_dim,
        "tempogram_dim": config.tempogram_dim,
        "pulse_dim": config.pulse_dim,
        "history_dim": config.history_dim,
        "candidate_dim": config.candidate_dim,
        "history_hidden": config.history_hidden,
        "encoder_width": config.encoder_width,
        "embedding_dim": config.embedding_dim,
        "interaction_rank": config.interaction_rank,
        "acoustic_stride": config.acoustic_stride,
        "acoustic_pool_kernel": config.acoustic_pool_kernel,
        "max_parameters": config.max_parameters,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    for name, value in (
        ("acoustic_kernel", config.acoustic_kernel),
        ("high_rate_kernel", config.high_rate_kernel),
        ("tempogram_kernel", config.tempogram_kernel),
        ("acoustic_pool_kernel", config.acoustic_pool_kernel),
    ):
        if value <= 0 or value % 2 == 0:
            raise ValueError(f"{name} must be a positive odd integer, got {value}")
    if config.acoustic_stride <= 0:
        raise ValueError("acoustic_stride must be positive")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError(f"dropout must be in [0,1), got {config.dropout}")
    for name, dilations in (
        ("acoustic_dilations", config.acoustic_dilations),
        ("high_rate_dilations", config.high_rate_dilations),
        ("tempogram_dilations", config.tempogram_dilations),
    ):
        if not dilations or any(dilation <= 0 for dilation in dilations):
            raise ValueError(f"{name} must contain positive integers")


def interpolate_encoded_sequence(
    sequence: torch.Tensor,
    times_ms: torch.Tensor,
    *,
    frame_origin_ms: float,
    frame_hop_ms: float,
    frame_valid: torch.Tensor | None = None,
    sequence_length: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly gather a fixed-bank encoded sequence at raw millisecond times.

    ``sequence_length`` limits lookup to the bank's valid prefix. When it is
    supplied, ``frame_valid`` describes that prefix rather than padded storage.
    """

    if sequence.ndim != 2 or sequence.shape[0] == 0:
        raise ValueError("sequence must have shape [time>0, channels]")
    if sequence_length is None:
        sequence_length = sequence.shape[0]
    elif (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or not 0 < sequence_length <= sequence.shape[0]
    ):
        raise ValueError("sequence_length must be a positive integer within sequence storage")
    if not torch.is_floating_point(times_ms):
        times_ms = times_ms.to(dtype=sequence.dtype)
    if not math.isfinite(frame_origin_ms):
        raise ValueError("frame_origin_ms must be finite")
    if not math.isfinite(frame_hop_ms) or frame_hop_ms <= 0.0:
        raise ValueError("frame_hop_ms must be positive and finite")
    if frame_valid is not None and (frame_valid.shape != (sequence_length,) or frame_valid.dtype != torch.bool):
        raise ValueError("frame_valid must be bool with shape [sequence_length]")

    positions = (times_ms - frame_origin_ms) / frame_hop_ms
    lower = torch.floor(positions).to(dtype=torch.long)
    exact = torch.isclose(positions, lower.to(dtype=positions.dtype), rtol=0.0, atol=1e-6)
    inside = (positions >= 0.0) & (positions <= sequence_length - 1)
    lower_safe = lower.clamp(0, sequence_length - 1)
    upper_safe = (lower_safe + 1).clamp_max(sequence_length - 1)
    alpha = (positions - lower_safe.to(dtype=positions.dtype)).clamp(0.0, 1.0)
    gathered = sequence[lower_safe] * (1.0 - alpha.unsqueeze(-1)) + sequence[upper_safe] * alpha.unsqueeze(-1)
    if frame_valid is not None:
        source_valid = frame_valid[lower_safe] & (frame_valid[upper_safe] | exact)
        inside &= source_valid
    return gathered.masked_fill(~inside.unsqueeze(-1), 0.0), inside


def triangular_support_log_scores(point_scores: torch.Tensor, *, half_width_ms: int) -> torch.Tensor:
    """Integrate the last score axis with the card's triangular support kernel."""

    if isinstance(half_width_ms, bool) or not isinstance(half_width_ms, int) or half_width_ms < 0:
        raise ValueError("half_width_ms must be a non-negative integer")
    expected = 2 * half_width_ms + 1
    if point_scores.ndim < 2 or point_scores.shape[-1] != expected:
        raise ValueError(f"point_scores must end in {expected} support offsets")
    offsets = torch.arange(-half_width_ms, half_width_ms + 1, device=point_scores.device)
    weights = (half_width_ms + 1 - offsets.abs()).to(dtype=point_scores.dtype)
    weights /= float((half_width_ms + 1) ** 2)
    return torch.logsumexp(point_scores + weights.log(), dim=-1)


def triangular_support_choice_nll(
    point_scores: torch.Tensor,
    *,
    half_width_ms: int,
    case_index: int = 0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Differentiable matched-choice NLL over candidate supports."""

    support_scores = triangular_support_log_scores(point_scores, half_width_ms=half_width_ms)
    if support_scores.ndim < 1 or not 0 <= case_index < support_scores.shape[-1]:
        raise ValueError("case_index is outside the candidate axis")
    losses = torch.logsumexp(support_scores, dim=-1) - support_scores[..., case_index]
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")
