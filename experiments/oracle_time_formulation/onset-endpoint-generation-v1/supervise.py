"""Bound one fresh typed-onset generation arm and retain failures verbatim."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil

ROOT=Path('artifacts/oracle-time-continuation/m3-20260917')
OWNER=ROOT/'onset-endpoint-generation-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def size(path):
    result=0
    for f in path.rglob('*'):
        try:
            if f.is_file():result+=f.stat().st_size
        except FileNotFoundError:
            pass  # Export staging can disappear between directory scan and stat.
    return result


def main():
    arm=sys.argv[1];assert arm in ('prior','context')
    out=ROOT/f'quality-onset-endpoint-{arm}-v1';assert not out.exists()
    command=[sys.executable,str(OWNER/'generate.py'),arm]
    save(OWNER/f'{arm}-launch.json',dict(command=command,script_sha256={p.name:digest(p) for p in OWNER.glob('*.py')},
        seconds_bound=1800,rss_bound=6*1024**3,min_available=2*1024**3,output_bound=2*1024**3))
    began=time.monotonic()
    with (OWNER/f'{arm}.log').open('x') as stream:
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(arm=arm,pid=child.pid,output=str(out))),flush=True)
        try:
            while child.poll() is None:
                p=psutil.Process(child.pid)
                assert time.monotonic()-began<1800,'wall-clock bound'
                assert sum(x.memory_info().rss for x in [p,*p.children(recursive=True)] if x.is_running())<6*1024**3,'RSS bound'
                assert psutil.virtual_memory().available>=2*1024**3,'available-memory guard'
                assert size(out)<2*1024**3 and shutil.disk_usage(ROOT).free>=512*1024**2,'disk guard'
                time.sleep(2)
            assert child.returncode==0,f'{arm} exited {child.returncode}'
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
    result=dict(status='complete',arm=arm,seconds=time.monotonic()-began,readout_sha256=digest(out/'readout.json'))
    save(OWNER/f'{arm}-status.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
