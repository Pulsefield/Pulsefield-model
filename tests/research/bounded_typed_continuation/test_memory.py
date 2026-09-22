import ctypes
import os
import platform
import re
import subprocess

import pytest

from ensomi_model.research.bounded_typed_continuation import memory


@pytest.mark.skipif(platform.system() != 'Darwin', reason='Darwin task-info ABI')
def test_task_footprint_matches_independent_vmmap_measurement():
    assert ctypes.sizeof(memory.TaskVMInfoRev1) == 152
    before = memory.footprint_bytes()
    value = subprocess.check_output(['vmmap', '-summary', str(os.getpid())], text=True)
    match = re.search(r'Physical footprint:\s+([0-9.]+)([KMG])', value)
    assert match is not None
    observed = float(match[1]) * 1024 ** ('KMG'.index(match[2]) + 1)
    after = memory.footprint_bytes()
    # vmmap rounds the display; subprocess creation also changes live pages.
    assert min(before, after) - 16 * 1024 ** 2 <= observed <= max(before, after) + 16 * 1024 ** 2
    assert before > 0


def test_non_darwin_does_not_load_mach_library(monkeypatch):
    monkeypatch.setattr(memory.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(memory, 'mach_library', lambda: pytest.fail('Mach is unavailable on Linux'))
    assert memory.footprint_bytes() is None
