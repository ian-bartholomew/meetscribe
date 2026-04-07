import threading
import multiprocessing
import multiprocessing.synchronize

# Patch multiprocessing.RLock to use threading.RLock instead.
# tqdm creates a multiprocessing.RLock which crashes in Textual worker
# threads on Python 3.13 due to fork_exec/resource_tracker issues.
# This must happen before any tqdm import (via faster_whisper, huggingface_hub, etc).
_real_rlock = threading.RLock


class _ThreadRLock(multiprocessing.synchronize.RLock):
    """Drop-in replacement that uses a thread lock instead of a semaphore."""
    def __init__(self, *args, **kwargs):
        # Skip the SemLock __init__ entirely
        self._lock = _real_rlock()

    def acquire(self, *args, **kwargs):
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        return self._lock.release()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *args):
        self._lock.release()


multiprocessing.RLock = _ThreadRLock

from meetscribe.tui.app import MeetscribeApp


def main() -> None:
    app = MeetscribeApp()
    app.run()


if __name__ == "__main__":
    main()
