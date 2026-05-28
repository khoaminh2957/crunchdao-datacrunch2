"""
download_data.py — wrapper around `crunch download`.

Pulls latest training + inference data for the active competition.
Must be run from inside the competition project directory (created by `crunch setup`).
"""

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    cwd = Path.cwd()
    if not (cwd / "main.py").exists() and not (cwd / "crunchdao.toml").exists():
        print(
            "[ERROR] This script must be run from inside a competition project directory.\n"
            "        Run `crunch setup <competition> <project>` first, then `cd <project>`.",
            file=sys.stderr,
        )
        return 2

    print(f"[download] cwd = {cwd}")
    print("[download] invoking `crunch download`...")

    try:
        result = subprocess.run(
            ["crunch", "download"],
            check=False,
            capture_output=False,
        )
    except FileNotFoundError:
        print(
            "[ERROR] `crunch` CLI not on PATH. Activate venv: .\\.venv\\Scripts\\Activate.ps1",
            file=sys.stderr,
        )
        return 3

    if result.returncode != 0:
        print(f"[download] crunch download exited with code {result.returncode}", file=sys.stderr)
        return result.returncode

    data_dir = cwd / "data"
    if data_dir.exists():
        files = sorted(data_dir.glob("*"))
        total_mb = sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
        print(f"[download] OK — {len(files)} files, {total_mb:.1f} MB in {data_dir}")
        for f in files[:10]:
            size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
            print(f"    {f.name:40s}  {size_mb:8.2f} MB")
        if len(files) > 10:
            print(f"    ... and {len(files) - 10} more")
    else:
        print("[download] WARNING: data/ directory not found after download", file=sys.stderr)
        return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
