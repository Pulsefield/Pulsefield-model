"""One fresh bounded subprocess, with process-tree cleanup on guard failure."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

from extract import OWNER, save
from pulsefield_model.research.oracle_time_continuation.storage import file_digest


def main():
    mode=sys.argv[1]
    assert mode in ('extract-rest','fit')
    destination=OWNER/('features-rest' if mode=='extract-rest' else 'fit')
    assert not destination.exists(), 'Inspect partial outputs; no blind restart or overwrite'
    limit=1800 if mode=='extract-rest' else 1200
    spent=json.loads((OWNER/'resume-input.json').read_text())['charged_extraction_seconds'] if mode=='extract-rest' else 0.
    script=Path(__file__).with_name('resume_extract.py' if mode=='extract-rest' else 'fit_resumed.py')
    save(OWNER/f'{mode}-launch.json',dict(command=[sys.executable,str(script)],seconds_bound=limit,
        scripts_sha256={p.name:file_digest(p) for p in sorted(script.parent.glob('*.py'))},
        rss_bound=6*1024**3,min_available=2*1024**3,owner_byte_bound=2*1024**3))
    started=time.monotonic()
    with (OWNER/f'{mode}.log').open('x') as output:
        child=subprocess.Popen([sys.executable,str(script)],stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(mode=mode,pid=child.pid,output=str(destination))),flush=True)
        try:
            while child.poll() is None:
                process=psutil.Process(child.pid)
                rss=sum(p.memory_info().rss for p in [process,*process.children(recursive=True)] if p.is_running())
                assert time.monotonic()-started<limit, 'wall-clock bound'
                if mode=='extract-rest':
                    assert spent+time.monotonic()-started<2700, 'aggregate extraction bound'
                assert rss<=6*1024**3, 'process-tree RSS bound'
                assert psutil.virtual_memory().available>=2*1024**3, 'available memory guard'
                assert sum(p.stat().st_size for p in OWNER.rglob('*') if p.is_file())<2*1024**3, 'owner size guard'
                time.sleep(2)
            assert child.returncode==0, f'{mode} exited {child.returncode}'
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
    result=dict(status='complete',mode=mode,seconds=time.monotonic()-started,
                readout_sha256=file_digest(destination/('manifest.json' if mode=='extract-rest' else 'readout.json')))
    save(OWNER/f'{mode}-status.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
