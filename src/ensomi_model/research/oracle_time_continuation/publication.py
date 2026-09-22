"""One recoverable staging directory per published file or source directory."""
from contextlib import contextmanager
from pathlib import Path
import shutil

from ..scoped_style_modeling.dataset import ContractError


@contextmanager
def staging_directory(destination: Path):
    """Lock a POSIX publication and reclaim only its previous incomplete staging.

    SIGKILL releases the kernel lock. A later writer removes the single owned
    staging directory before using it again, so interrupted retries cannot
    accumulate temporary checkpoints. The destination is untouched until the
    caller atomically replaces it. This lock covers publication, not run journals.
    """
    import fcntl

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name('.' + destination.name + '.staging')
    with destination.with_name('.' + destination.name + '.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError(f'Another writer owns publication: {destination}') from exc
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        try:
            yield staging
        finally:
            if staging.exists():
                shutil.rmtree(staging)
