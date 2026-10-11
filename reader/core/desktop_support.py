"""Small startup helpers, also used by isolated AI worker processes."""
import json
import os
import shutil
import sys
import uuid
from pathlib import Path


def python_command(executable: str | None = None) -> list[str]:
    command = [executable or sys.executable]
    if os.environ.get('AURIS_DESKTOP') == '1':
        command += ['-s', '-X', 'utf8']
    return command


def activate_gpu_runtime() -> None:
    root = os.environ.get('AURIS_DATA_DIR', '').strip()
    if not root:
        return
    runtime = Path(root) / 'runtime' / 'gpu'
    if (runtime / 'ready.json').is_file():
        sys.path.insert(0, str(runtime))


def install_pending_gpu_runtime(root: Path) -> None:
    """Swap in a GPU build staged while the previous one was loaded.

    Runs in the desktop process before anything imports torch; if the old
    folder is still locked, the swap is retried at the next start.
    """
    runtime = Path(root) / 'runtime'
    pending, target = runtime / 'gpu-pending', runtime / 'gpu'
    if not (pending / 'ready.json').is_file():
        return
    retired = runtime / ('gpu-old-' + uuid.uuid4().hex)
    try:
        if target.exists():
            target.rename(retired)
        pending.rename(target)
    except OSError:
        if retired.exists() and not target.exists():
            retired.rename(target)
        return
    shutil.rmtree(retired, ignore_errors=True)


class InstanceLock:
    """OS file lock: released on crashes, scoped to this data directory."""
    def __init__(self, root: Path):
        self.path = root / 'runtime' / 'desktop.lock'
        self.handle = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+b')
        self.handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            self.handle = None
            return False
        if self.path.stat().st_size == 0:
            self.handle.write(b'0')
            self.handle.flush()
        return True

    def close(self) -> None:
        if self.handle:
            self.handle.close()
            self.handle = None


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)
