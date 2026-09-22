"""macOS task footprint, including memory omitted by resident-set accounting."""
import ctypes
from functools import lru_cache
import platform

from ..scoped_style_modeling.dataset import ContractError


class TaskVMInfoRev1(ctypes.Structure):
    # mach/task_info.h: TASK_VM_INFO_REV1 ends at phys_footprint. The kernel
    # accepts this older, fixed-size prefix of the append-only VM-info ABI.
    _fields_ = [('virtual_size', ctypes.c_uint64), ('region_count', ctypes.c_int32),
                ('page_size', ctypes.c_int32)] + [(name, ctypes.c_uint64) for name in (
                    'resident_size', 'resident_size_peak', 'device', 'device_peak',
                    'internal', 'internal_peak', 'external', 'external_peak', 'reusable', 'reusable_peak',
                    'purgeable_volatile_pmap', 'purgeable_volatile_resident', 'purgeable_volatile_virtual',
                    'compressed', 'compressed_peak', 'compressed_lifetime', 'phys_footprint')]


@lru_cache(maxsize=1)
def mach_library():
    library = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
    library.task_info.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
    library.task_info.restype = ctypes.c_int
    return library


def footprint_bytes():
    """Read this process's task footprint on Darwin; unavailable elsewhere."""
    if platform.system() != 'Darwin':
        return None
    library = mach_library()
    task = ctypes.c_uint32.in_dll(library, 'mach_task_self_').value
    value = TaskVMInfoRev1()
    expected = ctypes.sizeof(value) // ctypes.sizeof(ctypes.c_uint32)
    count = ctypes.c_uint32(expected)
    result = library.task_info(task, 22, ctypes.byref(value), ctypes.byref(count))
    if result != 0 or count.value < expected:
        raise ContractError(f'macOS task footprint unavailable: task_info={result}, count={count.value}')
    return int(value.phys_footprint)
