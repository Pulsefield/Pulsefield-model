from io import BytesIO
import json
import shutil
import wave

import numpy as np
import pytest

from pulsefield_model.research.vacation_training.audio import cache_asset, frame_chunks, mel_frames, run_audio
from pulsefield_model.research.vacation_training.control import publish_json
from pulsefield_model.research.vacation_training.config import AudioConfig
from pulsefield_model.research.oracle_time_continuation.storage import file_digest
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError


def test_streaming_frame_grid_matches_whole_waveform_across_short_reads():
    values = np.random.default_rng(4).normal(size=12001).astype('<f4')
    class ShortReads(BytesIO):
        def read(self, count=-1):
            return super().read(min(count, 19))
    pieces = list(frame_chunks(ShortReads(values.tobytes()), 7))
    assert [first for first, _ in pieces] == list(range(0, 46, 7))
    result = np.concatenate([mel_frames(samples) for _, samples in pieces])
    expected = mel_frames(values)
    assert result.shape == expected.shape == (46, 128)
    np.testing.assert_allclose(result, expected, atol=.002, rtol=.001)
    with pytest.raises(ContractError, match='incomplete'):
        list(frame_chunks(BytesIO(b'x'), 7))


class Control:
    def __init__(self, stop_after=None):
        self.calls, self.stop_after = 0, stop_after
    def boundary(self):
        self.calls += 1
        return 'pause_file' if self.stop_after and self.calls > self.stop_after else None
    def storage(self, additional=0):
        pass


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='requires FFmpeg')
def test_real_codec_resume_keeps_identical_frames_and_rejects_corrupt_shards(tmp_path):
    path = tmp_path / 'audio.wav'
    with wave.open(str(path), 'wb') as stream:
        stream.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        values = (np.sin(np.arange(13000) * .12) * 14000).astype('<i2')
        stream.writeframes(values.tobytes())
    asset = dict(path=str(path), sha256=file_digest(path), sources=[dict(source_sha256='a'*64,
        split='train', group_id='group', first_ms=0., last_ms=400.)])
    cfg = AudioConfig(chunk_frames=7)
    reference, reason = cache_asset(asset, tmp_path / 'reference', cfg, Control())
    assert reason is None and reference['complete']
    partial, reason = cache_asset(asset, tmp_path / 'resume', cfg, Control(1))
    assert reason == 'pause_file' and not partial['complete'] and len(partial['shards']) == 1
    resumed, reason = cache_asset(asset, tmp_path / 'resume', cfg, Control())
    assert resumed == reference and reason is None
    for shard in reference['shards']:
        assert (tmp_path / 'reference' / shard['file']).read_bytes() == (tmp_path / 'resume' / shard['file']).read_bytes()
    bad = tmp_path / 'resume' / reference['shards'][0]['file']
    bad.write_bytes(b'corrupt')
    with pytest.raises(ContractError, match='digest'):
        cache_asset(asset, tmp_path / 'resume', cfg, Control())


def test_missing_inventory_and_cross_split_assets_are_reported_without_payload_reads(tmp_path):
    path = tmp_path / 'inputs.json'
    value = dict(format='vacation/audio-inputs-v1', assets=[], issues=[dict(reason='missing_audio_filename')])
    sha = publish_json(path, value)
    result = run_audio(AudioConfig(manifest_file=str(path), manifest_sha256=sha), tmp_path / 'missing', Control())
    assert result['status'] == 'completed_with_anomalies' and result['completed'] == 0
    value['assets'] = [dict(path='does-not-exist.mp3', sha256='a'*64, sources=[dict(
        source_sha256='b'*64, group_id='validation', split='validation', first_ms=0, last_ms=1000)])]
    sha = publish_json(path, value)
    result = run_audio(AudioConfig(manifest_file=str(path), manifest_sha256=sha), tmp_path / 'quarantined', Control())
    assert result['quarantined'] == 1 and result['failed'] == 0
