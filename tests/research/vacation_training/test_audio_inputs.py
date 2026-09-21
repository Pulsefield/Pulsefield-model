from dataclasses import asdict
import json

from pulsefield_model.research.bounded_typed_continuation.corpus import PLAN_FORMAT, SamplingConfig, draw_plan
from pulsefield_model.research.oracle_time_continuation.storage import SourceStore, file_digest
from pulsefield_model.research.vacation_training.audio_inputs import prepare_audio_inputs
from pulsefield_model.research.vacation_training.control import publish_json


def test_train_inventory_deduplicates_audio_and_only_reads_train_payloads(tmp_path):
    audio = tmp_path / 'shared.wav'
    audio.write_bytes(b'content identity is independent of the later decoder audit')
    store = SourceStore(tmp_path / 'cache')
    sources, catalog = {}, []
    for i in range(2):
        text = ('osu file format v14\n[General]\nMode:3\nAudioFilename:shared.wav\n'
                f'[Metadata]\nTitle:chart {i}\n[Difficulty]\nCircleSize:4\n[HitObjects]\n')
        text += ''.join(f'{64+128*(n%4)},192,{n*100},1,0,0:0:0:0:\n' for n in range(70))
        path = tmp_path / f'{i}.osu'
        path.write_text(text)
        sha = file_digest(path)
        source = store.admit(path, sha, group_id='train-group', split='train')
        metadata = json.loads((store.root / sha / 'metadata.json').read_text())
        sources[sha] = dict(identity=asdict(source.identity), rows=70, seed_rows=30, onsets=40,
            rows_sha256=metadata['rows_sha256'], metadata_sha256=file_digest(store.root / sha / 'metadata.json'))
        catalog.append(dict(**asdict(source.identity), path=path.name, event_count=70))
    # VAL metadata in the same directory is an ambiguity flag, not permission
    # to open its absent payload or claim a cross-split content comparison.
    catalog.append(dict(source_sha256='f'*64, split='validation', path='absent-validation.osu', group_id='val'))
    catalog_path = tmp_path / 'catalog.json'
    catalog_sha = publish_json(catalog_path, catalog)
    sampling = SamplingConfig(horizons=(8,), milestones=(16,))
    plan = dict(format=PLAN_FORMAT, catalog_sha256=catalog_sha, split_sha256='a'*64, census_sha256='b'*64,
                sources=sources, sampling=asdict(sampling), draws=draw_plan(sources, sampling))
    plan_file = tmp_path / 'plan.json'
    plan_sha = publish_json(plan_file, plan)
    output = tmp_path / 'audio-inputs.json'
    result = prepare_audio_inputs(plan_file=plan_file, plan_sha256=plan_sha, catalog_file=catalog_path,
        catalog_sha256=catalog_sha, catalog_root=tmp_path, source_cache_dir=store.root, output_file=output)
    assert result['assets'] == 1 and result['missing_or_ambiguous'] == 0
    asset, = json.loads(output.read_text())['assets']
    assert asset['sha256'] == file_digest(audio) and len(asset['sources']) == 2
    assert asset['known_directory_splits'] == ['train', 'validation']
