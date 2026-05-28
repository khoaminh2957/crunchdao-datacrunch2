"""
leaderboard_track.py — fetch user rank for the active competition and append to CSV.

Strategy:
  1. Try `crunch leaderboard --me` CLI subcommand (if available in installed version).
  2. Fall back to direct API call against api.crunchdao.com using CRUNCHDAO_API_KEY.
  3. Append (timestamp, competition, rank, score, n_participants) to leaderboard_history.csv.

The CLI's leaderboard surface has changed between versions, so we hedge.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "leaderboard_history.csv"
API_BASE = "https://api.hub.crunchdao.com"


def _try_cli() -> Optional[dict]:
    """Try the CLI first — quietest path, no API plumbing needed."""
    for sub in (["leaderboard", "--me", "--json"], ["leaderboard", "me", "--json"]):
        try:
            result = subprocess.run(
                ["crunch"] + sub,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    continue
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _try_api() -> Optional[dict]:
    """Direct API call. Endpoint shape inferred from public crunch-cli source.

    If endpoints have changed, this will log the response and return None so
    the caller logs an unknown rank rather than crashing.
    """
    api_key = os.environ.get("CRUNCHDAO_API_KEY")
    if not api_key:
        print("[track] CRUNCHDAO_API_KEY not in env — cannot call API. Set it in .env", file=sys.stderr)
        return None

    headers = {"Authorization": f"API-Key {api_key}"}

    # Pull the user's active competitions/submissions
    try:
        r = requests.get(f"{API_BASE}/v1/users/me/submissions", headers=headers, timeout=30)
    except requests.RequestException as e:
        print(f"[track] API request failed: {e}", file=sys.stderr)
        return None

    if r.status_code != 200:
        print(f"[track] API returned {r.status_code}: {r.text[:300]}", file=sys.stderr)
        return None

    try:
        data = r.json()
    except json.JSONDecodeError:
        print(f"[track] non-JSON API response: {r.text[:300]}", file=sys.stderr)
        return None

    # Best-effort: most recent submission
    subs = data if isinstance(data, list) else data.get("submissions", [])
    if not subs:
        print("[track] no submissions found via API", file=sys.stderr)
        return None

    latest = max(subs, key=lambda s: s.get("createdAt", s.get("created_at", "")))
    return {
        "competition": latest.get("competition", {}).get("name") if isinstance(latest.get("competition"), dict) else latest.get("competition"),
        "rank": latest.get("rank"),
        "score": latest.get("score") or latest.get("metric"),
        "n_participants": latest.get("totalParticipants") or latest.get("total_participants"),
        "submission_id": latest.get("id"),
    }


def main() -> int:
    info = _try_cli()
    source = "cli"
    if info is None:
        info = _try_api()
        source = "api"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    row = {
        "timestamp": ts,
        "source": source if info else "none",
        "competition": (info or {}).get("competition") or "unknown",
        "rank": (info or {}).get("rank"),
        "score": (info or {}).get("score"),
        "n_participants": (info or {}).get("n_participants"),
        "submission_id": (info or {}).get("submission_id"),
    }

    # Append to CSV
    new_file = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)

    if info is None:
        print(f"[track] could not fetch rank (CLI + API both failed). Row logged as unknown.")
        return 1

    rank = row["rank"]
    n = row["n_participants"]
    score = row["score"]
    pct = f"{(rank / n * 100):.1f}%" if (rank and n) else "?"
    print(f"[track] {row['competition']}: rank {rank}/{n} ({pct}), score={score} — appended to {CSV_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
