from __future__ import annotations

import copy
import json
import math
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pulsefield_model.evals.mir_anchor_probe import (
    MIRProbeRunConfig,
    MIRProbeRunReport,
    PROBE_COALITIONS,
    _encoder_bank_capacity,
    _load_probe_songs,
    aggregate_mir_probe_runs,
    erode_valid_mask,
    extract_mir_anchor_features,
    prepare_mir_anchor_manifest,
    run_mir_anchor_probe,
)
from pulsefield_model.features.mir_backbone import MIRBackboneConfig, MIRProbeFeatures
from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects


@pytest.mark.parametrize(
    "field",
    (
        "train_choice_sets_per_batch",
        "eval_choice_sets_per_batch",
        "encoder_chunk_frames",
        "encoder_max_fast_frames",
    ),
)
@pytest.mark.parametrize("value", (0, -1, True))
def test_probe_runtime_rejects_invalid_memory_bound(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        MIRProbeRunConfig(**{field: value})


def test_encoder_capacity_uses_a_bounded_power_of_two_bucket_set() -> None:
    from pulsefield_model.evals.mir_anchor_model import MirAnchorProbeConfig

    run_config = MIRProbeRunConfig(
        encoder_chunk_frames=128,
        encoder_max_fast_frames=1_024,
    )
    model_config = MirAnchorProbeConfig()
    feature_config = MIRBackboneConfig()

    capacities = {
        _encoder_bank_capacity(
            "N",
            actual_length=length,
            audio_id="fixture",
            run_config=run_config,
            model_config=model_config,
            feature_config=feature_config,
        )
        for length in range(1, 1_025)
    }
    assert capacities == {128, 256, 512, 1_024}
    with pytest.raises(ValueError, match="increase encoder_max_fast_frames"):
        _encoder_bank_capacity(
            "N",
            actual_length=1_025,
            audio_id="fixture",
            run_config=run_config,
            model_config=model_config,
            feature_config=feature_config,
        )

    production_config = MIRProbeRunConfig()
    assert _encoder_bank_capacity(
        "N",
        actual_length=131_073,
        audio_id="fixture",
        run_config=production_config,
        model_config=model_config,
        feature_config=feature_config,
    ) == 147_456
    for group in ("A", "T"):
        assert _encoder_bank_capacity(
            group,
            actual_length=32_769,
            audio_id="fixture",
            run_config=production_config,
            model_config=model_config,
            feature_config=feature_config,
        ) == 40_960
        with pytest.raises(ValueError, match="configured limit 36864"):
            _encoder_bank_capacity(
                group,
                actual_length=36_865,
                audio_id="fixture",
                run_config=production_config,
                model_config=model_config,
                feature_config=feature_config,
            )


def test_prepare_selects_one_chart_per_audio_and_splits_at_audio_level(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    index_rows: list[dict[str, object]] = []
    for index in range(20):
        beatmap_set = dataset_root / "0" / str(1_000 + index)
        beatmap_set.mkdir(parents=True)
        audio_path = beatmap_set / "song.ogg"
        audio_path.write_bytes(b"synthetic audio")
        easier_path = beatmap_set / "a.osu"
        harder_path = beatmap_set / "b.osu"
        _write_osu(easier_path)
        _write_osu(harder_path)
        for beatmap_path, difficulty, version in (
            (easier_path, 3.5, "easier"),
            (harder_path, 4.5, "harder"),
        ):
            index_rows.append(
                {
                    "shard": "0",
                    "audio_path": f"{1_000 + index}/song.ogg",
                    "beatmap_path": f"{1_000 + index}/{beatmap_path.name}",
                    "beatmap_set_id": 1_000 + index,
                    "audio_lead_in": index,
                    "title": f"Song {index}",
                    "artist": "Fixture",
                    "creator": "Tests",
                    "version": version,
                    "difficulty": difficulty,
                }
            )

    source_index = pd.DataFrame(index_rows)
    first_path = tmp_path / "first.parquet"
    second_path = tmp_path / "second.parquet"
    loader = lambda _: source_index.copy()  # noqa: E731
    first = prepare_mir_anchor_manifest(
        index_path=tmp_path / "unused.parquet",
        dataset_root=dataset_root,
        output_path=first_path,
        audio_count=20,
        seed=73,
        controls_per_case=1,
        index_loader=loader,
    )
    second = prepare_mir_anchor_manifest(
        index_path=tmp_path / "unused.parquet",
        dataset_root=dataset_root,
        output_path=second_path,
        audio_count=20,
        seed=73,
        controls_per_case=1,
        index_loader=loader,
    )

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first, pd.read_parquet(first_path), check_dtype=False)
    assert first["audio_id"].tolist() == [f"audio_{index:05d}" for index in range(20)]
    assert first["audio_group"].nunique() == 20
    assert first["audio_path"].nunique() == 20
    assert first["beatmap_path"].str.endswith("/a.osu").all()
    assert first["version"].eq("easier").all()
    assert first["audio_lead_in"].sort_values().tolist() == list(range(20))
    assert first["difficulty"].eq(3.5).all()
    assert first["split"].value_counts().to_dict() == {"train": 14, "validation": 3, "test": 3}
    assert first["row_count"].eq(7).all()
    assert first["episode_count"].eq(6).all()
    assert first["structural_choice_count"].gt(0).all()


def test_prepare_samples_requested_audio_count_deterministically(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    rows = []
    for index in range(8):
        beatmap_set = dataset_root / "0" / str(index)
        beatmap_set.mkdir(parents=True)
        (beatmap_set / "song.ogg").write_bytes(b"audio")
        _write_osu(beatmap_set / "map.osu")
        rows.append(
            {
                "shard": "0",
                "audio_path": f"{index}/song.ogg",
                "beatmap_path": f"{index}/map.osu",
                "difficulty": 4.0,
                "artist": "Fixture",
                "title": f"Song {index}",
            }
        )
    source_index = pd.DataFrame(rows)

    def run(name: str, seed: int) -> pd.DataFrame:
        return prepare_mir_anchor_manifest(
            index_path=tmp_path / "unused.parquet",
            dataset_root=dataset_root,
            output_path=tmp_path / f"{name}.parquet",
            audio_count=4,
            seed=seed,
            controls_per_case=1,
            index_loader=lambda _: source_index.copy(),
        )

    first = run("first", 11)
    repeated = run("repeated", 11)
    other_seed = run("other", 12)

    assert first["audio_path"].tolist() == repeated["audio_path"].tolist()
    assert first["audio_path"].tolist() != other_seed["audio_path"].tolist()


def test_prepare_uses_exact_requested_split_counts(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    rows = []
    for index in range(10):
        beatmap_set = dataset_root / "0" / str(index)
        beatmap_set.mkdir(parents=True)
        (beatmap_set / "song.ogg").write_bytes(b"audio")
        _write_osu(beatmap_set / "map.osu")
        rows.append(
            {
                "shard": "0",
                "audio_path": f"{index}/song.ogg",
                "beatmap_path": f"{index}/map.osu",
                "difficulty": 4.0,
                "artist": "Fixture",
                "title": f"Song {index}",
            },
        )

    manifest = prepare_mir_anchor_manifest(
        index_path=tmp_path / "unused.parquet",
        dataset_root=dataset_root,
        output_path=tmp_path / "manifest.parquet",
        audio_count=10,
        split_counts={"train": 6, "validation": 2, "test": 2},
        seed=19,
        controls_per_case=1,
        index_loader=lambda _: pd.DataFrame(rows),
    )

    assert manifest["split"].value_counts().to_dict() == {
        "train": 6,
        "validation": 2,
        "test": 2,
    }


def test_prepare_deduplicates_normalized_artist_title_groups(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    rows = []
    for index, (artist, title) in enumerate(
        (("Artist", "Same Song"), (" ARTIST ", "same  song"), ("Other", "Different")),
    ):
        beatmap_set = dataset_root / "0" / str(index)
        beatmap_set.mkdir(parents=True)
        (beatmap_set / "song.ogg").write_bytes(f"audio {index}".encode())
        _write_osu(beatmap_set / "map.osu")
        rows.append(
            {
                "shard": "0",
                "audio_path": f"{index}/song.ogg",
                "beatmap_path": f"{index}/map.osu",
                "difficulty": 4.0,
                "artist": artist,
                "title": title,
            },
        )

    manifest = prepare_mir_anchor_manifest(
        index_path=tmp_path / "unused.parquet",
        dataset_root=dataset_root,
        output_path=tmp_path / "manifest.parquet",
        audio_count=2,
        seed=3,
        controls_per_case=1,
        index_loader=lambda _path: pd.DataFrame(rows),
    )

    assert manifest["audio_group"].nunique() == 2
    assert manifest["artist"].str.strip().str.casefold().eq("artist").sum() == 1


def test_extract_writes_uncompressed_probe_groups_and_skips_valid_files(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.parquet"
    audio_paths = [tmp_path / "one.ogg", tmp_path / "two.ogg"]
    for audio_path in audio_paths:
        audio_path.write_bytes(b"audio fixture")
    pd.DataFrame(
        {
            "audio_id": ["audio_00000", "audio_00001"],
            "audio_path": [str(path) for path in audio_paths],
        }
    ).to_parquet(manifest_path, index=False)

    loaded: list[tuple[Path, int]] = []
    extracted: list[tuple[int, int]] = []

    def audio_loader(path: Path, sample_rate: int) -> np.ndarray:
        loaded.append((path, sample_rate))
        return np.arange(12, dtype=np.float32)

    def feature_extractor(
        waveform: object,
        sample_rate: int,
        config: MIRBackboneConfig,
    ) -> MIRProbeFeatures:
        extracted.append((np.asarray(waveform).size, sample_rate))
        return _probe_features(config)

    output_dir = tmp_path / "features"
    first = extract_mir_anchor_features(
        manifest_path=manifest_path,
        output_dir=output_dir,
        audio_loader=audio_loader,
        feature_extractor=feature_extractor,
    )
    second = extract_mir_anchor_features(
        manifest_path=manifest_path,
        output_dir=output_dir,
        audio_loader=audio_loader,
        feature_extractor=feature_extractor,
    )
    changed_config = extract_mir_anchor_features(
        manifest_path=manifest_path,
        output_dir=output_dir,
        config=MIRBackboneConfig(novelty_clip=3.0),
        audio_loader=audio_loader,
        feature_extractor=feature_extractor,
    )

    assert first.audio_count == 2
    assert first.written_count == 2
    assert first.skipped_count == 0
    assert second.written_count == 0
    assert second.skipped_count == 2
    assert changed_config.written_count == 2
    assert changed_config.skipped_count == 0
    assert loaded == [
        (audio_paths[0], 24_000),
        (audio_paths[1], 24_000),
        (audio_paths[0], 24_000),
        (audio_paths[1], 24_000),
    ]
    assert extracted == [(12, 24_000)] * 4

    stored_path = output_dir / "audio_00000.npz"
    with np.load(stored_path, allow_pickle=False) as stored:
        assert set(stored.files) == {
            "fast_frame_centers_s",
            "slow_frame_centers_s",
            "A",
            "N",
            "T",
            "P",
            "A_valid",
            "N_valid",
            "T_valid",
            "P_valid",
            "audio_duration_ms",
            "config_json",
            "feature_schema",
            "source_audio_path",
        }
        assert stored["A"].shape == (4, 128)
        assert stored["N"].shape == (4, 5)
        assert stored["T"].shape == (3, 122)
        assert stored["P"].shape == (4, 6)
        assert stored["audio_duration_ms"] == 0.5
    with zipfile.ZipFile(stored_path) as archive:
        assert all(member.compress_type == zipfile.ZIP_STORED for member in archive.infolist())


def test_extract_replaces_only_an_invalid_existing_file(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.parquet"
    pd.DataFrame(
        {
            "audio_id": ["audio_00000", "audio_00001"],
            "audio_path": [str(tmp_path / "one.ogg"), str(tmp_path / "two.ogg")],
        }
    ).to_parquet(manifest_path, index=False)
    output_dir = tmp_path / "features"
    extractor_calls = 0

    def extract(_: object, __: int, config: MIRBackboneConfig) -> MIRProbeFeatures:
        nonlocal extractor_calls
        extractor_calls += 1
        return _probe_features(config)

    arguments = {
        "manifest_path": manifest_path,
        "output_dir": output_dir,
        "audio_loader": lambda _path, _sample_rate: np.zeros(1, dtype=np.float32),
        "feature_extractor": extract,
    }
    extract_mir_anchor_features(**arguments)
    np.savez(output_dir / "audio_00001.npz", A=np.zeros((1, 1), dtype=np.float32))
    report = extract_mir_anchor_features(**arguments)

    assert extractor_calls == 3
    assert report.written_count == 1
    assert report.skipped_count == 1


def test_extract_replaces_sequential_cache_id_when_manifest_audio_changes(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.parquet"
    first_audio = tmp_path / "first.ogg"
    second_audio = tmp_path / "second.ogg"
    loaded: list[Path] = []

    def write_manifest(audio_path: Path) -> None:
        pd.DataFrame({"audio_id": ["audio_00000"], "audio_path": [str(audio_path)]}).to_parquet(
            manifest_path,
            index=False,
        )

    arguments = {
        "manifest_path": manifest_path,
        "output_dir": tmp_path / "features",
        "audio_loader": lambda path, _sample_rate: (loaded.append(path) or np.zeros(12, dtype=np.float32)),
        "feature_extractor": lambda _waveform, _sample_rate, config: _probe_features(config),
    }
    write_manifest(first_audio)
    first = extract_mir_anchor_features(**arguments)
    write_manifest(second_audio)
    second = extract_mir_anchor_features(**arguments)

    assert first.written_count == second.written_count == 1
    assert loaded == [first_audio.resolve(), second_audio.resolve()]
    with np.load(tmp_path / "features" / "audio_00000.npz", allow_pickle=False) as stored:
        assert stored["source_audio_path"].item() == second_audio.resolve().as_posix()


def test_mask_erosion_requires_the_complete_encoder_receptive_field() -> None:
    valid = np.array([True, True, True, False, True, True, True], dtype=np.bool_)

    assert np.array_equal(erode_valid_mask(valid, radius_frames=0), valid)
    assert np.array_equal(
        erode_valid_mask(valid, radius_frames=1),
        np.array([False, True, False, False, False, True, False]),
    )
    assert not erode_valid_mask(np.ones(4, dtype=np.bool_), radius_frames=2).any()


def test_probe_loading_rejects_nonfinite_durable_features(tmp_path: Path) -> None:
    from pulsefield_model.evals.mir_anchor_model import MirAnchorProbeConfig

    audio_path = tmp_path / "audio.ogg"
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    feature_path = feature_dir / "audio_00000.npz"
    _write_probe_cache(feature_path, seed=0, source_audio_path=audio_path)
    with np.load(feature_path, allow_pickle=False) as stored:
        payload = {name: np.array(stored[name], copy=True) for name in stored.files}
    payload["A"][20, 0] = np.nan
    np.savez(feature_path, **payload)
    manifest = pd.DataFrame(
        {
            "audio_id": ["audio_00000"],
            "audio_group": ["fixture:audio"],
            "split": ["train"],
            "audio_path": [str(audio_path)],
            "beatmap_path": [str(tmp_path / "map.osu")],
        },
    )

    with pytest.raises(ValueError, match="finite floating-point"):
        _load_probe_songs(
            manifest,
            feature_dir=feature_dir,
            run_config=MIRProbeRunConfig(controls_per_case=1),
            model_config=MirAnchorProbeConfig(acoustic_dim=8, tempogram_dim=28),
            hitobject_loader=lambda _path: (),
        )


def test_probe_loading_skips_songs_over_the_encoder_memory_bound(tmp_path: Path) -> None:
    from pulsefield_model.evals.mir_anchor_model import MirAnchorProbeConfig

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    manifest_rows = []
    for index, fast_frame_count in enumerate((1_025, 601)):
        audio_id = f"audio_{index:05d}"
        audio_path = tmp_path / f"audio_{index}.ogg"
        beatmap_path = tmp_path / f"map_{index}.osu"
        _write_osu(beatmap_path)
        _write_probe_cache(
            feature_dir / f"{audio_id}.npz",
            seed=index,
            source_audio_path=audio_path,
            fast_frame_count=fast_frame_count,
        )
        manifest_rows.append(
            {
                "audio_id": audio_id,
                "audio_group": f"fixture:{audio_id}",
                "split": "train",
                "audio_path": str(audio_path),
                "beatmap_path": str(beatmap_path),
            }
        )

    with pytest.warns(RuntimeWarning, match="audio_00000"):
        loaded = _load_probe_songs(
            pd.DataFrame(manifest_rows),
            feature_dir=feature_dir,
            run_config=MIRProbeRunConfig(
                encoder_max_fast_frames=1_024,
                controls_per_case=1,
                support_half_width_ms=1,
                history_rows=4,
            ),
            model_config=MirAnchorProbeConfig(
                acoustic_dim=8,
                tempogram_dim=28,
                acoustic_dilations=(1,),
                high_rate_dilations=(1,),
                tempogram_dilations=(1,),
            ),
            hitobject_loader=parse_mania_hit_objects,
        )

    assert tuple(song.audio_id for song in loaded) == ("audio_00001",)


def test_probe_stage_builds_exact_supports_trains_all_coalitions_and_reports_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from pulsefield_model.evals import mir_anchor_probe as probe_module
    from pulsefield_model.evals.mir_anchor_model import MirAnchorProbeConfig

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    manifest_rows = []
    for index, split in enumerate(("train", "validation", "test")):
        audio_id = f"audio_{index:05d}"
        beatmap_path = tmp_path / f"map_{index}.osu"
        audio_path = tmp_path / f"audio_{index}.ogg"
        _write_osu(beatmap_path)
        _write_probe_cache(
            feature_dir / f"{audio_id}.npz",
            seed=index,
            source_audio_path=audio_path,
        )
        manifest_rows.append(
            {
                "audio_id": audio_id,
                "audio_group": f"fixture:{audio_id}",
                "split": split,
                "audio_path": str(audio_path),
                "beatmap_path": str(beatmap_path),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    run_config = MIRProbeRunConfig(
        epochs=1,
        max_choice_sets_per_song=8,
        train_choice_sets_per_batch=3,
        eval_choice_sets_per_batch=2,
        encoder_chunk_frames=128,
        encoder_max_fast_frames=1_024,
        controls_per_case=1,
        support_half_width_ms=1,
        history_rows=4,
        seed=19,
        device="cpu",
    )
    model_config = MirAnchorProbeConfig(
        acoustic_dim=8,
        tempogram_dim=28,
        history_hidden=4,
        encoder_width=4,
        embedding_dim=4,
        interaction_rank=2,
        acoustic_dilations=(1,),
        high_rate_dilations=(1,),
        tempogram_dilations=(1,),
        dropout=0.0,
    )

    loaded = _load_probe_songs(
        manifest,
        feature_dir=feature_dir,
        run_config=run_config,
        model_config=model_config,
        hitobject_loader=parse_mania_hit_objects,
    )
    assert len(loaded) == 3
    assert loaded[0].candidate_center_times_ms.ndim == 2
    assert not hasattr(loaded[0], "features")
    assert loaded[0].feature_path.name == "audio_00000.npz"
    other_model_seed = _load_probe_songs(
        manifest,
        feature_dir=feature_dir,
        run_config=replace(run_config, seed=20),
        model_config=model_config,
        hitobject_loader=parse_mania_hit_objects,
    )
    assert np.array_equal(loaded[0].episode_indices, other_model_seed[0].episode_indices)

    from pulsefield_model.evals.mir_anchor_model import MirAnchorProbe

    torch.manual_seed(run_config.seed)
    unbatched_model = MirAnchorProbe(model_config)
    batched_model = copy.deepcopy(unbatched_model)
    training_indexes = np.arange(
        min(run_config.max_choice_sets_per_song, loaded[0].candidate_center_times_ms.shape[0]),
        dtype=np.int64,
    )

    def accumulated_objective_and_gradients(model, batch_size):
        encoded = probe_module._encode_audio_groups(
            model,
            loaded[0],
            run_config=run_config,
            device=torch.device("cpu"),
        )
        result = probe_module._backward_probe_choice_batches(
            model,
            loaded[0],
            encoded,
            training_indexes,
            batch_size=batch_size,
            run_config=run_config,
            device=torch.device("cpu"),
        )
        assert result is not None
        detached_sum, valid_count = result
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(float(valid_count))
        return detached_sum / valid_count

    unbatched_objective = accumulated_objective_and_gradients(
        unbatched_model,
        training_indexes.size,
    )
    batched_objective = accumulated_objective_and_gradients(
        batched_model,
        run_config.train_choice_sets_per_batch,
    )
    torch.testing.assert_close(unbatched_objective, batched_objective, rtol=1e-6, atol=2e-7)
    for (name, unbatched_parameter), batched_parameter in zip(
        unbatched_model.named_parameters(),
        batched_model.parameters(),
        strict=True,
    ):
        assert unbatched_parameter.grad is not None, name
        assert batched_parameter.grad is not None, name
        torch.testing.assert_close(
            unbatched_parameter.grad,
            batched_parameter.grad,
            rtol=2e-4,
            atol=2e-6,
        )

    original_validation_nll = probe_module._validation_nll

    def checked_validation_nll(model, songs, *, run_config, device):
        assert all(parameter.grad is None for parameter in model.parameters())
        return original_validation_nll(
            model,
            songs,
            run_config=run_config,
            device=device,
        )

    monkeypatch.setattr(probe_module, "_validation_nll", checked_validation_nll)

    report = run_mir_anchor_probe(
        manifest_path=manifest_path,
        feature_dir=feature_dir,
        output_dir=tmp_path / "probe",
        run_config=run_config,
        model_config=model_config,
    )
    reference_report = run_mir_anchor_probe(
        manifest_path=manifest_path,
        feature_dir=feature_dir,
        output_dir=tmp_path / "probe_unbatched_reference",
        run_config=replace(
            run_config,
            train_choice_sets_per_batch=run_config.max_choice_sets_per_song,
            eval_choice_sets_per_batch=run_config.max_choice_sets_per_song,
        ),
        model_config=model_config,
    )

    pd.testing.assert_frame_equal(
        pd.read_parquet(report.per_audio_path),
        pd.read_parquet(reference_report.per_audio_path),
        check_exact=False,
        rtol=2e-4,
        atol=2e-6,
    )

    assert report.seed == 19
    assert report.parameter_count > 0
    assert report.best_epoch == 0
    assert np.isfinite(report.best_validation_nll)
    assert report.effective_audio_counts == {"train": 1, "validation": 1, "test": 1}
    assert all(count > 0 for count in report.effective_case_counts.values())
    assert report.state_path.is_file()
    per_audio = pd.read_parquet(report.per_audio_path)
    observed = per_audio[per_audio["condition"] == "observed"]
    assert set(observed["coalition"]) == {
        "H" if not coalition else "H+" + "+".join(coalition)
        for coalition in PROBE_COALITIONS
    }
    assert observed.groupby("audio_id").size().eq(9).all()
    assert {
        "N_shift_reference",
        "N_circular_shift",
        "T_shift_reference",
        "T_circular_shift",
        "P_shift_reference",
        "P_circular_shift",
        "NTP_shift_reference",
        "NTP_circular_shift",
    }.issubset(per_audio["condition"])
    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    assert summary["alignment_sensitivity"] == {
        "within_song_mir_circular_shift": ["N", "T", "P", "NTP"],
    }
    assert summary["effective_audio_counts"] == {"test": 1, "train": 1, "validation": 1}
    assert summary["coverage"]["test"]["eligible_choice_count"] > 0
    assert "jointly trained" in summary["scientific_scope"]["estimand"]
    assert summary["evaluation"]["test"]["observed"]["H"]["audio_count"] == 1
    assert "mir_over_acoustic" in summary["evaluation"]["test"]["paired_effects"]
    assert set(summary["evaluation"]["test"]["mir_shapley"]) == {"N", "T", "P"}


def test_multi_seed_inference_averages_each_audio_before_bootstrap(tmp_path: Path) -> None:
    reports = []
    full_coalition = "H+A+N+P+T"
    coalition_losses = {
        "H": 2.0,
        "H+A": 1.8,
        "H+A+N": 1.7,
        "H+A+P": 1.7,
        "H+A+T": 1.7,
        "H+A+N+P": 1.6,
        "H+A+N+T": 1.6,
        "H+A+P+T": 1.6,
        full_coalition: 1.5,
    }
    for seed in (3, 5):
        rows = []
        for audio_position, audio_id in enumerate(("audio_a", "audio_b")):
            offset = 0.1 * seed + 0.2 * audio_position
            for coalition, loss in coalition_losses.items():
                rows.append(
                    {
                        "audio_id": audio_id,
                        "split": "test",
                        "condition": "observed",
                        "coalition": coalition,
                        "conditional_nll": loss + offset,
                        "episode_count": 20,
                        "eligible_choice_count": 10,
                        "choice_count": 10,
                    },
                )
            for group in ("N", "T", "P", "NTP"):
                rows.extend(
                    (
                        {
                            "audio_id": audio_id,
                            "split": "test",
                            "condition": f"{group}_shift_reference",
                            "coalition": full_coalition,
                            "conditional_nll": 1.5 + offset,
                            "episode_count": 20,
                            "eligible_choice_count": 10,
                            "choice_count": 8,
                        },
                        {
                            "audio_id": audio_id,
                            "split": "test",
                            "condition": f"{group}_circular_shift",
                            "coalition": full_coalition,
                            "conditional_nll": 1.7 + offset,
                            "episode_count": 20,
                            "eligible_choice_count": 10,
                            "choice_count": 8,
                        },
                    ),
                )
        run_dir = tmp_path / f"seed_{seed}"
        run_dir.mkdir()
        per_audio_path = run_dir / "per_audio.parquet"
        pd.DataFrame(rows).to_parquet(per_audio_path, index=False)
        summary_path = run_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "run_config": {"seed": seed, "controls_per_case": 16},
                    "model_config": {"embedding_dim": 32},
                    "feature_config": {"mel_hop_ms": 5},
                    "data_sources": {
                        "audio_a": {
                            "audio_path": "/audio/a.ogg",
                            "beatmap_path": "/beatmaps/a.osu",
                        },
                        "audio_b": {
                            "audio_path": "/audio/b.ogg",
                            "beatmap_path": "/beatmaps/b.osu",
                        },
                    },
                },
            ),
            encoding="utf-8",
        )
        reports.append(
            MIRProbeRunReport(
                seed=seed,
                parameter_count=1,
                best_epoch=0,
                best_validation_nll=1.0,
                effective_audio_counts={"test": 2},
                effective_case_counts={"test": 2},
                state_path=run_dir / "state.pt",
                per_audio_path=per_audio_path,
                summary_path=summary_path,
            ),
        )

    output_path = tmp_path / "multi_seed_inference.json"
    result = aggregate_mir_probe_runs(reports, output_path=output_path, bootstrap_seed=17)

    assert result["seeds"] == [3, 5]
    assert result["seed_selection"] is False
    assert result["primary_metric"] == "mir_over_acoustic"
    assert {
        "audio_over_history",
        "mir_over_acoustic",
        "shapley_N",
        "shapley_T",
        "shapley_P",
    }.issubset(result["metrics"])
    assert result["metrics"]["mir_over_acoustic"]["mean_nll_reduction"] == pytest.approx(0.3)
    assert result["metrics"]["mir_over_acoustic"]["audio_count"] == 2
    assert result["metrics"]["mir_over_acoustic"]["between_seed_sd"] == pytest.approx(0.0)
    assert "one_sided_sign_flip_p" in result["metrics"]["mir_over_acoustic"]
    assert "one_sided_sign_flip_p" not in result["metrics"]["shapley_N"]
    assert result["unavailable_metrics"] == []
    assert output_path.is_file()
    per_audio = pd.read_parquet(result["per_audio_path"])
    assert per_audio.groupby("metric")["audio_id"].nunique().eq(2).all()


def test_multi_seed_inference_rejects_different_experiment_metadata(tmp_path: Path) -> None:
    per_audio_path = tmp_path / "per_audio.parquet"
    pd.DataFrame(
        {
            "audio_id": ["audio_a"],
            "split": ["test"],
            "condition": ["observed"],
            "coalition": ["H"],
            "conditional_nll": [1.0],
            "episode_count": [4],
            "eligible_choice_count": [2],
            "choice_count": [2],
        },
    ).to_parquet(per_audio_path, index=False)
    reports = []
    for seed, source in ((1, "/audio/a.ogg"), (2, "/audio/other.ogg")):
        summary_path = tmp_path / f"summary_{seed}.json"
        summary_path.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "run_config": {"seed": seed, "controls_per_case": 16},
                    "model_config": {},
                    "feature_config": {},
                    "data_sources": {
                        "audio_a": {
                            "audio_path": source,
                            "beatmap_path": "/beatmaps/a.osu",
                        },
                    },
                },
            ),
            encoding="utf-8",
        )
        reports.append(
            MIRProbeRunReport(
                seed=seed,
                parameter_count=1,
                best_epoch=0,
                best_validation_nll=1.0,
                effective_audio_counts={"test": 1},
                effective_case_counts={"test": 2},
                state_path=tmp_path / f"state_{seed}.pt",
                per_audio_path=per_audio_path,
                summary_path=summary_path,
            ),
        )

    with pytest.raises(ValueError, match="different data or experiment configs"):
        aggregate_mir_probe_runs(reports, output_path=tmp_path / "aggregate.json")


def _write_osu(path: Path) -> None:
    lines = [
        "osu file format v14",
        "",
        "[General]",
        "Mode:3",
        "",
        "[Difficulty]",
        "CircleSize:4",
        "",
        "[HitObjects]",
    ]
    for row_index, time_ms in enumerate((0, 100, 600, 700, 1_400, 1_500, 2_400)):
        if row_index == 0:
            lines.append(f"64,192,{time_ms},128,0,80:0:0:0:0:")
        else:
            lines.append(f"64,192,{time_ms},1,0,0:0:0:0:")
        if row_index == 0:
            lines.append(f"192,192,{time_ms},1,0,0:0:0:0:")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _probe_features(config: MIRBackboneConfig) -> MIRProbeFeatures:
    fast_centers = np.arange(4, dtype=np.float64) * config.mel_hop_seconds
    slow_centers = np.arange(3, dtype=np.float64) * (config.tempogram_hop_ms / 1_000.0)
    return MIRProbeFeatures(
        fast_frame_centers_s=fast_centers,
        slow_frame_centers_s=slow_centers,
        acoustic=np.full((4, config.mel_bins), 1.0, dtype=np.float32),
        novelty=np.full((4, 5), 2.0, dtype=np.float32),
        tempogram=np.full((3, config.tempo_bins + 26), 3.0, dtype=np.float32),
        pulse=np.full((4, 6), 4.0, dtype=np.float32),
        acoustic_valid=np.ones(4, dtype=np.bool_),
        novelty_valid=np.array([False, True, True, True], dtype=np.bool_),
        tempogram_valid=np.array([False, True, False], dtype=np.bool_),
        pulse_valid=np.array([False, True, True, False], dtype=np.bool_),
    )


def _write_probe_cache(
    path: Path,
    *,
    seed: int,
    source_audio_path: Path,
    fast_frame_count: int = 601,
) -> None:
    rng = np.random.default_rng(seed)
    feature_config = MIRBackboneConfig(mel_bins=8, tempo_bins=2)
    slow_frame_count = math.ceil(fast_frame_count / 4)
    fast_centers_s = np.arange(fast_frame_count, dtype=np.float64) * 0.005
    slow_centers_s = np.arange(slow_frame_count, dtype=np.float64) * 0.020
    np.savez(
        path,
        fast_frame_centers_s=fast_centers_s,
        slow_frame_centers_s=slow_centers_s,
        A=rng.normal(size=(fast_frame_count, 8)).astype(np.float32),
        N=rng.normal(size=(fast_frame_count, 5)).astype(np.float32),
        T=rng.normal(size=(slow_frame_count, 28)).astype(np.float32),
        P=rng.normal(size=(fast_frame_count, 6)).astype(np.float32),
        A_valid=np.ones(fast_frame_count, dtype=np.bool_),
        N_valid=np.ones(fast_frame_count, dtype=np.bool_),
        T_valid=np.ones(slow_frame_count, dtype=np.bool_),
        P_valid=np.ones(fast_frame_count, dtype=np.bool_),
        audio_duration_ms=np.asarray(3_000.0, dtype=np.float64),
        config_json=np.asarray(json.dumps(asdict(feature_config), sort_keys=True, separators=(",", ":"))),
        feature_schema=np.asarray("mir_anchor_v1"),
        source_audio_path=np.asarray(source_audio_path.resolve().as_posix()),
    )
