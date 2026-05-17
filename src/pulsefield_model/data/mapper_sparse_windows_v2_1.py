from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch

from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects
from pulsefield_model.data.control_windows import FRAME_HOP_MS, normalize_difficulty
from pulsefield_model.data.mapper_tuple_windows import (
    MapperTupleWindowDataset,
    MapperTupleWindowRecord,
    collate_mapper_tuple_windows,
    control_teacher_cache_key,
    extract_mapper_density_8s,
    is_mapper_tuple_window_start_allowed,
    load_control_teacher_cache_entry,
)
from pulsefield_model.models.mapper.v2_1.replay import NO_EMITTED_LANE_INDEX, ln_carry_state_tensors
from pulsefield_model.models.mapper.v2_1.tokenizer import (
    MAPPER_WRITE_MS,
    TokenizedMapperWindow,
    UnsupportedMapperActionError as MapperV21UnsupportedMapperActionError,
    encode_mapper_window,
    hitobjects_to_mapper_timepoints,
    mapper_chart_end_ms,
)
from pulsefield_model.models.mapper.shared.tokenizer import UnsupportedMapperActionError as MapperTupleUnsupportedMapperActionError
from pulsefield_model.models.mapper.v2_1.vocab import MapperV21Vocab


MAPPER_V21_RECORD_CACHE_SCHEMA_VERSION = 2
MAPPER_V21_TOKENIZER_CACHE_VERSION = 2
MAPPER_WRITE_FRAMES = MAPPER_WRITE_MS // FRAME_HOP_MS
MAPPER_CONTEXT_FRAMES = MAPPER_WRITE_FRAMES


class MapperV21WindowDataset(MapperTupleWindowDataset):
    """Mapper V2.1 sparse-token 8s window dataset.

    The source control windows, full-song context, and optional control-teacher
    cache layout are shared with Mapper tuple/V2. The tokenization contract is not:
    v2.1 uses sparse lane tokens and carries same-time lane replay state.
    """

    def __init__(
        self,
        *args: Any,
        vocab: MapperV21Vocab | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_vocab = MapperV21Vocab() if vocab is None else vocab
        if not isinstance(resolved_vocab, MapperV21Vocab):
            raise TypeError(f"vocab must be a MapperV21Vocab, got {type(resolved_vocab).__name__}")
        super().__init__(*args, vocab=resolved_vocab, **kwargs)

    def _record_cache_validity_metadata(self) -> dict[str, Any]:
        metadata = super()._record_cache_validity_metadata()
        metadata.update(
            {
                "schema_version": MAPPER_V21_RECORD_CACHE_SCHEMA_VERSION,
                "tokenizer_cache_version": MAPPER_V21_TOKENIZER_CACHE_VERSION,
                "mapper_token_contract": "v2.1_sparse_lane_actions",
                "vocab_size": int(self.vocab.size),
                "time_shift_values_ms": list(self.vocab.time_shift_values_ms),
            }
        )
        return metadata

    def __getitem__(self, index: int) -> dict[str, Any]:
        mapper_record = self.records[index]
        record = mapper_record.control_record
        tokenized = self._tokenize_record(record)
        cache_path = self.control_teacher_cache_path(record)
        cache_entry = None
        if cache_path is not None and cache_path.exists():
            cache_entry = load_control_teacher_cache_entry(cache_path, record=record)
        elif self.require_control_teacher_cache and cache_path is not None:
            raise FileNotFoundError(f"missing mapper v2.1 control teacher cache entry: {cache_path}")
        elif self.require_control_teacher_cache:
            raise ValueError("require_control_teacher_cache=True requires control_teacher_cache_dir")

        metadata = {
            "beatmap_path": record.beatmap_path.as_posix(),
            "audio_path": record.audio_path.as_posix(),
            "difficulty": record.difficulty,
            "source_frame_count": record.frame_count,
            "inference_frame_count": mapper_v2_1_padded_frame_count(record),
            "target_start_frame": record.target_start_frame,
            "target_start_ms": record.target_start_ms,
            "chart_end_ms": tokenized.chart_end_ms,
            "control_record_index": mapper_record.control_record_index,
            "mapper_token_contract": "v2.1_sparse_lane_actions",
        }
        if cache_path is not None:
            metadata["control_teacher_cache_key"] = control_teacher_cache_key(record)
            metadata["control_teacher_cache_path"] = cache_path.as_posix()
            metadata["control_teacher_cache_hit"] = cache_entry is not None

        sample: dict[str, Any] = {
            "difficulty": torch.tensor([record.difficulty], dtype=torch.float32),
            "normalized_difficulty": torch.tensor([normalize_difficulty(record.difficulty)], dtype=torch.float32),
            "decoder_input_tokens": tokenized.decoder_input_tensor(),
            "target_fragment_tokens": tokenized.target_fragment_tensor(),
            "target_fragment_states": {
                "current_ms": tokenized.target_fragment_current_ms,
                "open_mask": tokenized.target_fragment_open_mask,
                "open_start_ms": tokenized.target_fragment_open_start_ms,
                "open_age_ms": tokenized.target_fragment_open_age_ms,
                "emitted_lane_mask": tokenized.target_fragment_emitted_lane_mask,
                "last_lane_index": tokenized.target_fragment_last_lane_index,
            },
            "ln_carry_in": ln_carry_state_tensors(tokenized.ln_carry_in),
            "ln_carry_out": ln_carry_state_tensors(tokenized.ln_carry_out),
            "close_labels": tokenized.close_labels,
            "close_label_mask": tokenized.close_label_mask,
            "write_start_ms": torch.tensor(tokenized.write_start_ms, dtype=torch.long),
            "write_end_ms": torch.tensor(tokenized.write_end_ms, dtype=torch.long),
            "chart_end_ms": torch.tensor(tokenized.chart_end_ms, dtype=torch.long),
            "is_full_chart_start": torch.tensor(tokenized.is_full_chart_start, dtype=torch.bool),
            "is_full_chart_end": torch.tensor(tokenized.is_full_chart_end, dtype=torch.bool),
            "metadata": metadata,
        }
        density_target_8s, density_confidence_8s = extract_mapper_density_8s(
            self._load_control_v3_target_8s(record),
        )
        if cache_entry is not None:
            sample["control_memory_8s"] = cache_entry["control_memory_8s"]
            sample["density_teacher_8s"] = cache_entry["density_teacher_8s"]
            sample["density_target_8s"] = density_target_8s
            sample["density_confidence_8s"] = density_confidence_8s
            if self.include_full_song_context:
                sample.update(self._load_full_song_context_fields(mapper_record, record))
            return sample

        sample.update(self._load_full_song_context_fields(mapper_record, record))
        sample["density_target_8s"] = density_target_8s
        sample["density_confidence_8s"] = density_confidence_8s
        return sample

    def _tokenize_record(self, record: Any) -> TokenizedMapperWindow:
        write_start_ms = int(record.target_start_ms)
        write_end_ms = write_start_ms + MAPPER_WRITE_MS
        try:
            timepoints = self._load_timepoints(record.beatmap_path)
            chart_end_ms = mapper_chart_end_ms(timepoints)
            if chart_end_ms < write_start_ms:
                raise MapperTupleUnsupportedMapperActionError(
                    f"mapper v2.1 write window starts after chart_end_ms: {write_start_ms} > {chart_end_ms}",
                )
            return encode_mapper_window(
                timepoints,
                vocab=self.vocab,
                write_start_ms=write_start_ms,
                write_end_ms=write_end_ms,
                chart_start_ms=0,
                chart_end_ms=chart_end_ms,
            )
        except MapperV21UnsupportedMapperActionError as exc:
            raise MapperTupleUnsupportedMapperActionError(str(exc)) from exc

    def _load_timepoints(self, beatmap_path: Path) -> tuple:
        key = beatmap_path.as_posix()
        cached = self._timepoints_by_beatmap.get(key)
        if cached is not None:
            self._timepoints_by_beatmap.move_to_end(key)
            return cached
        cached = tuple(hitobjects_to_mapper_timepoints(parse_mania_hit_objects(beatmap_path, expected_key_count=4)))
        if self.max_cached_timepoint_maps > 0:
            self._timepoints_by_beatmap[key] = cached
            while len(self._timepoints_by_beatmap) > self.max_cached_timepoint_maps:
                self._timepoints_by_beatmap.popitem(last=False)
        return cached


def collate_mapper_v2_1_windows(samples: Sequence[dict[str, Any]], *, pad_id: int = 0) -> dict[str, Any]:
    if not samples:
        raise ValueError("collate_mapper_v2_1_windows requires at least one sample")
    batch = collate_mapper_tuple_windows(samples, pad_id=pad_id)
    batch_size, max_seq_len = batch["target_fragment_tokens"].shape
    emitted_lane_mask = torch.zeros((batch_size, max_seq_len, 4), dtype=torch.bool)
    last_lane_index = torch.full((batch_size, max_seq_len), NO_EMITTED_LANE_INDEX, dtype=torch.long)

    for batch_index, sample in enumerate(samples):
        length = int(sample["target_fragment_tokens"].shape[0])
        states = sample["target_fragment_states"]
        if "emitted_lane_mask" not in states or "last_lane_index" not in states:
            raise ValueError("mapper v2.1 samples must include emitted_lane_mask and last_lane_index states")
        emitted = states["emitted_lane_mask"].to(dtype=torch.bool)
        last = states["last_lane_index"].to(dtype=torch.long)
        if tuple(emitted.shape) != (length, 4):
            raise ValueError(f"sample {batch_index} emitted_lane_mask must have shape {(length, 4)}")
        if tuple(last.shape) != (length,):
            raise ValueError(f"sample {batch_index} last_lane_index must have shape {(length,)}")
        emitted_lane_mask[batch_index, :length] = emitted
        last_lane_index[batch_index, :length] = last

    batch["target_fragment_states"] = dict(batch["target_fragment_states"])
    batch["target_fragment_states"]["emitted_lane_mask"] = emitted_lane_mask
    batch["target_fragment_states"]["last_lane_index"] = last_lane_index
    return batch


def is_mapper_v2_1_window_start_allowed(
    record: Any,
    *,
    mapper_stride_frames: int = MAPPER_WRITE_FRAMES,
) -> bool:
    return is_mapper_tuple_window_start_allowed(record, mapper_stride_frames=mapper_stride_frames)


def mapper_v2_1_padded_frame_count(record: Any) -> int:
    return max(int(record.frame_count), int(record.target_start_frame) + MAPPER_WRITE_FRAMES)
