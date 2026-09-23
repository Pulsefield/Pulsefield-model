from dataclasses import asdict

import numpy as np
import torch

from ensomi_model.research.bounded_typed_continuation.features import content_features
from ensomi_model.research.joint_audio_continuation.batching import collate, interpolate_audio, score_batch
from ensomi_model.research.joint_audio_continuation.data import JointChart, query
from ensomi_model.research.joint_audio_continuation.model import JointAudioModel, JointModelConfig
from ensomi_model.research.joint_audio_continuation.state import exact_features, legal_rows
from ensomi_model.research.bounded_typed_continuation.data import SourceChart
from ensomi_model.research.oracle_time_continuation.data import SourceIdentity
from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit
from ensomi_model.research.oracle_time_continuation.storage import ROW_DTYPE


def source_chart(release_ms):
    # The hold starts before the 511-row crop and remains open at every query.
    rows = np.zeros(533, dtype=ROW_DTYPE)
    rows['time'] = [*range(0, 5310, 10), release_ms, 8000]
    rows['actions'][:531, 1] = 1
    rows['actions'][0, 0] = 2
    rows['actions'][531, 0] = 3
    rows['actions'][532, 2] = 1
    identity = SourceIdentity('a' * 64, 'b' * 64, 'parity-group', 'train')
    source = SourceChart(identity, rows)
    mel = np.random.default_rng(723).normal(size=(900, 128)).astype(np.float32)
    return JointChart(asdict(identity), source, mel, 9000)


def test_long_prefix_training_matches_native_cache_without_future_ln_endpoints():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(918)
        model = JointAudioModel(JointModelConfig()).eval()
        assert model.temporal.config.receptive_tokens == 511
        a, b = source_chart(7000), source_chart(7500)
        samples = [query(chart, 5290, horizon_ms=317) for chart in (a, b)]
        assert all(sample.truncated and sample.source_index == 530 for sample in samples)
        assert all(len(sample.raw) == 511 and sample.predecessor_time_ms == 180 for sample in samples)
        assert samples[0].replay == samples[1].replay
        assert samples[0].replay.open_ln_start_ms[0] == 0.
        np.testing.assert_array_equal(samples[0].raw, samples[1].raw)

        with torch.no_grad():
            model.set_audio_normalization(torch.linspace(-2., 2., 128), torch.linspace(.3, 2., 128))
            for block in model.audio_blocks:
                block.norm.bias.fill_(.4)
            batch = collate(samples, [a, b], model.config)
            trained = score_batch(model, batch.inputs)
            torch.testing.assert_close(trained.timing_logits[0], trained.timing_logits[1], atol=0, rtol=0)
            torch.testing.assert_close(trained.row_log_probs[0], trained.row_log_probs[1], atol=0, rtol=0)

            # Reproduce rollout's append path from BOS, without source snapshots
            # or future endpoints. Its cached history is longer than the crop.
            replay, cache = ExactReplayState(), model.temporal.empty_cache()
            for index in range(530):
                row = a.source.row(index)
                previous = None if replay.last_row is None else replay.last_row.time_ms
                replay = commit(replay, row)
                raw = content_features([row], [previous], [[None] * 4])[0]
                cache = model.temporal.append(cache, torch.from_numpy(raw))
            assert replay == samples[0].replay and cache.rows == 530
            assert replay.open_ln_start_ms[0] == 0. and not replay.is_complete
            history = model.temporal.read(cache)[None]
            dense_history = model.encode_history(batch.inputs.raw, batch.inputs.history_valid,
                                                 batch.inputs.truncated)
            torch.testing.assert_close(history[0], dense_history[0], atol=2e-5, rtol=2e-5)

            encoded = model.encode_audio(torch.from_numpy(a.mel)[None])
            bin_times = torch.arange(529, 561) * 10 + 9
            torch.testing.assert_close(bin_times, batch.inputs.timing_times_ms[0])
            exact = torch.from_numpy(exact_features([replay] * len(bin_times), bin_times.tolist()))[None]
            timing = model.timing_logits(interpolate_audio(encoded, bin_times[None]), history, exact).flatten(1)
            row_time = samples[0].target_time_ms
            assert row_time == 5300
            row_exact = torch.from_numpy(exact_features([replay], [row_time]))
            row_legal = torch.from_numpy(legal_rows([replay], [False]))
            row = model.row_log_probs(interpolate_audio(encoded, torch.tensor([row_time])), history,
                                      row_exact, row_legal, torch.tensor([replay.occupancy]))
            torch.testing.assert_close(timing[0], trained.timing_logits[0], atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(row[0], trained.row_log_probs[0], atol=3e-5, rtol=3e-5)
    finally:
        torch.set_num_threads(previous_threads)
