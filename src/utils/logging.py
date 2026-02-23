"""Logging utilities shared across entry-point scripts."""
import sys
from pathlib import Path


class TeeLogger:
    """Duplicate stdout/stderr to a log file while still printing to console.

    Usage::

        tee = TeeLogger("outputs/logs/run.log")
        print("visible in both console and file")
        tee.close()            # restore original streams
    """

    def __init__(self, log_path: str):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, data: str):
        self._stdout.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._file.close()
