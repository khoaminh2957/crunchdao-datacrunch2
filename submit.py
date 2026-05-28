"""
submit.py — local validation then push.

Flow:
  1. crunch test  (runs your code in a sandbox identical to the grader)
  2. crunch push  (only if test passes and --dry-run not set)

Usage:
  python submit.py                # validate + push
  python submit.py --dry-run      # validate only, do NOT push
  python submit.py --message "v2: added rolling features"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str], desc: str) -> int:
    print(f"\n[submit] === {desc} ===")
    print(f"[submit] $ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print(f"[ERROR] `{cmd[0]}` not on PATH. Activate venv first.", file=sys.stderr)
        return 127
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run `crunch test` only, skip push")
    parser.add_argument("--message", default=None, help="Submission message for the leaderboard")
    parser.add_argument("--skip-test", action="store_true", help="Skip `crunch test` (NOT recommended)")
    args = parser.parse_args()

    cwd = Path.cwd()
    has_main = (cwd / "main.py").exists()
    has_toml = (cwd / "crunchdao.toml").exists()
    if not (has_main or has_toml):
        print(
            "[ERROR] No main.py or crunchdao.toml in cwd.\n"
            "        Run this from inside the competition project (created by `crunch setup`).",
            file=sys.stderr,
        )
        return 2

    # Step 1: local validation
    if not args.skip_test:
        rc = _run(["crunch", "test"], "crunch test (local sandbox validation)")
        if rc != 0:
            print(f"\n[submit] FAIL — crunch test returned {rc}. Fix errors above before pushing.", file=sys.stderr)
            return rc
        print("[submit] crunch test PASSED")
    else:
        print("[submit] --skip-test set, bypassing local validation (risky)")

    if args.dry_run:
        print("\n[submit] --dry-run set, NOT pushing. Done.")
        return 0

    # Step 2: push
    push_cmd = ["crunch", "push"]
    if args.message:
        push_cmd.extend(["--message", args.message])
    else:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        push_cmd.extend(["--message", f"baseline LGB+Ridge {ts}"])

    rc = _run(push_cmd, "crunch push (live submission)")
    if rc != 0:
        print(f"\n[submit] FAIL — crunch push returned {rc}.", file=sys.stderr)
        return rc

    print("\n[submit] SUCCESS — submission pushed. Check leaderboard at hub.crunchdao.com")
    print("[submit] Run leaderboard_track.py in ~30 min to log your rank.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
