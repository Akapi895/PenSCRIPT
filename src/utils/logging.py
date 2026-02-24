"""Logging utilities shared across entry-point scripts.

Provides :class:`TeeLogger` which mirrors *all* output to a log file while
optionally suppressing noisy environment-level lines from the console so the
terminal stays focused on training progress and summary metrics.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Default console-suppress patterns
# ---------------------------------------------------------------------------
# Lines matching ANY of these regexes are written to the **log file only** and
# hidden from the console.  They cover the per-host action prints and stdlib
# warnings emitted by the PenGym / NASim environment layer.
ENV_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s+Host\s.+Action\s'"),   # "  Host X Action 'scan' SUCCESS/FAILURE …"
    re.compile(r"^WARNING:root:"),           # stdlib logging.warning() from env layer
    re.compile(r"^None in not"),             # host_vector.py msfrpc edge case
    re.compile(r"^\*\s*WARNING:"),           # host_vector.py "* WARNING:" prints
    re.compile(r"^Over time:"),              # host_vector.py timeout message
    re.compile(r"^Shell for host"),          # host_vector.py shell-creation failure
]


class TeeLogger:
    """Duplicate stdout/stderr to a log file with optional console filtering.

    All output is **always** written to the log file.  Lines matching any
    pattern in *console_suppress* are hidden from the terminal.

    Usage::

        tee = TeeLogger("outputs/logs/run.log",
                         console_suppress=ENV_NOISE_PATTERNS)
        print("visible in both console and file")
        tee.close()            # restore original streams
    """

    def __init__(
        self,
        log_path: str,
        console_suppress: list[re.Pattern[str]] | None = None,
    ):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._suppress = console_suppress or []
        sys.stdout = self
        sys.stderr = self

    # ── core I/O ──────────────────────────────────────────────────

    def write(self, data: str) -> int:
        """Write *data* to the log file (always) and to the console (filtered).

        Returns the number of characters written (to satisfy the TextIO
        interface expected by ``sys.stdout``).
        """
        # Always persist to disk
        self._file.write(data)
        self._file.flush()

        if not self._suppress or not data:
            self._stdout.write(data)
            return len(data)

        # Line-level filtering for the console
        parts: list[str] = []
        for line in data.splitlines(keepends=True):
            if not any(p.search(line) for p in self._suppress):
                parts.append(line)
        if parts:
            self._stdout.write("".join(parts))
        return len(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        """Restore original stdout/stderr and close the log file."""
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._file.close()
