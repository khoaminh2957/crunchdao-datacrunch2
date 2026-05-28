#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dom_screenshot_daemon.py

Connects to Chrome via CDP on port 9222 and every 60s:
  - Navigates / reloads 3 CrunchDAO hub URLs
  - Captures full-page PNG screenshot
  - Extracts document.body.innerText
  - Detects new submissions / run status changes and appends to events log
Loops forever; gracefully retries connection failures.
"""

import os
import sys
import io
import json
import time
import base64
import datetime
import traceback
from urllib.request import urlopen
from urllib.error import URLError

import websocket  # pip install websocket-client

# ---------- UTF-8 stdout ----------
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------- Config ----------
CDP_HOST = "127.0.0.1"
CDP_PORT = 9222
WS_KW = dict(origin="http://localhost", suppress_origin=False)

BASE_DIR = r"C:\Users\Admin\earn5usd\crunchdao"
SCREENSHOT_DIR = os.path.join(BASE_DIR, "dom_screenshots")
TEXT_DIR = os.path.join(BASE_DIR, "dom_text")
EVENTS_LOG = os.path.join(BASE_DIR, "dom_events.log")

URLS = [
    ("submissions",
     "https://hub.crunchdao.com/competitions/datacrunch-2/projects/12867/curly-crayfishwetminh-v2/submissions"),
    ("leaderboard",
     "https://hub.crunchdao.com/competitions/datacrunch-2/leaderboard?projectName=curly-crayfishwetminh-v2"),
    ("runs",
     "https://hub.crunchdao.com/competitions/datacrunch-2/projects/12867/curly-crayfishwetminh-v2/runs"),
]

POLL_INTERVAL_S = 60
NAV_WAIT_S = 8          # wait after navigate for SPA to render
CONNECT_RETRY_S = 10
ERROR_RETRY_S = 30

# ---------- Globals for change detection ----------
_last_submission_ids = set()
_last_run_status = {}   # run_id -> status string
_msg_id = 0


def _ensure_dirs():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)


def _now_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _log_event(msg: str):
    line = f"[{_now_iso()}] {msg}\n"
    print(line, end="", flush=True)
    try:
        with open(EVENTS_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[{_now_iso()}] failed to write events log: {e}", flush=True)


# ---------- CDP helpers ----------
def _get_page_target():
    """Pick a 'page' target from /json; raise on failure."""
    url = f"http://{CDP_HOST}:{CDP_PORT}/json"
    with urlopen(url, timeout=5) as r:
        targets = json.loads(r.read().decode("utf-8"))
    pages = [t for t in targets if t.get("type") == "page"]
    if not pages:
        raise RuntimeError("No 'page' targets in /json")
    # Prefer a hub.crunchdao.com tab if one exists, else first page
    for t in pages:
        if "hub.crunchdao.com" in (t.get("url") or ""):
            return t
    return pages[0]


def _wait_for_cdp():
    """Block until CDP is reachable and has at least one page target."""
    while True:
        try:
            t = _get_page_target()
            _log_event(f"CDP reachable. Using target id={t.get('id')} url={t.get('url')}")
            return t
        except (URLError, ConnectionError, OSError, RuntimeError) as e:
            print(f"[{_now_iso()}] CDP not ready ({e}); sleeping {CONNECT_RETRY_S}s", flush=True)
            time.sleep(CONNECT_RETRY_S)


def _next_id():
    global _msg_id
    _msg_id += 1
    return _msg_id


def _send(ws, method, params=None, timeout=30):
    mid = _next_id()
    payload = {"id": mid, "method": method}
    if params is not None:
        payload["params"] = params
    ws.send(json.dumps(payload))
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = max(0.1, deadline - time.time())
        ws.settimeout(remaining)
        raw = ws.recv()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        # Skip event notifications without an id
        if msg.get("id") == mid:
            if "error" in msg:
                raise RuntimeError(f"CDP error on {method}: {msg['error']}")
            return msg.get("result", {})
    raise TimeoutError(f"Timeout waiting for CDP response to {method}")


def _connect_ws(target):
    ws_url = target["webSocketDebuggerUrl"]
    ws = websocket.create_connection(ws_url, **WS_KW)
    ws.settimeout(30)
    # Enable required domains
    _send(ws, "Page.enable")
    _send(ws, "Runtime.enable")
    _send(ws, "Network.enable")
    return ws


def _navigate_and_wait(ws, url):
    """Navigate (or reload if same URL) and wait briefly for render."""
    try:
        current = _send(ws, "Runtime.evaluate",
                        {"expression": "window.location.href", "returnByValue": True})
        current_url = (current.get("result") or {}).get("value", "")
    except Exception:
        current_url = ""

    if current_url.split("#")[0] == url.split("#")[0]:
        _send(ws, "Page.reload", {"ignoreCache": True})
    else:
        _send(ws, "Page.navigate", {"url": url})

    # Wait for SPA to render. Best-effort: poll document.readyState.
    deadline = time.time() + NAV_WAIT_S
    while time.time() < deadline:
        try:
            r = _send(ws, "Runtime.evaluate",
                      {"expression": "document.readyState", "returnByValue": True},
                      timeout=5)
            state = (r.get("result") or {}).get("value", "")
            if state == "complete":
                break
        except Exception:
            pass
        time.sleep(0.5)
    # Extra settle for client-side rendering
    time.sleep(2.0)


def _capture_screenshot(ws, slug):
    res = _send(ws, "Page.captureScreenshot",
                {"format": "png", "captureBeyondViewport": True}, timeout=60)
    data = res.get("data", "")
    if not data:
        return None
    raw = base64.b64decode(data)
    path = os.path.join(SCREENSHOT_DIR, f"{_now_str()}_{slug}.png")
    with open(path, "wb") as f:
        f.write(raw)
    return path


def _capture_text(ws, slug):
    expr = "document.body ? document.body.innerText : ''"
    res = _send(ws, "Runtime.evaluate",
                {"expression": expr, "returnByValue": True}, timeout=30)
    txt = (res.get("result") or {}).get("value", "") or ""
    path = os.path.join(TEXT_DIR, f"{_now_str()}_{slug}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    return path, txt


# ---------- Change detection ----------
def _detect_submission_changes(text: str):
    """Heuristic: find tokens that look like submission ids (e.g. #12345 or
    'Submission 12345'). Returns set of detected ids."""
    import re
    ids = set()
    for m in re.finditer(r"(?:Submission|submission|#)\s*#?(\d{3,8})", text):
        ids.add(m.group(1))
    return ids


def _detect_run_statuses(text: str):
    """Heuristic: pair (run_id, status) where run id is a 4-7 digit number
    appearing near a status word. Returns dict run_id -> status."""
    import re
    statuses = {}
    status_pat = re.compile(
        r"(\d{4,7}).{0,80}?\b(running|queued|completed|succeeded|success|failed|error|cancelled|canceled|pending)\b",
        re.IGNORECASE | re.DOTALL,
    )
    for m in status_pat.finditer(text):
        rid = m.group(1)
        st = m.group(2).lower()
        statuses[rid] = st
    return statuses


def _process_changes(slug: str, text: str):
    global _last_submission_ids, _last_run_status
    if slug == "submissions":
        ids = _detect_submission_changes(text)
        new_ids = ids - _last_submission_ids
        if _last_submission_ids and new_ids:
            for nid in sorted(new_ids):
                _log_event(f"NEW_SUBMISSION id={nid}")
        if ids:
            _last_submission_ids = ids
    elif slug == "runs":
        cur = _detect_run_statuses(text)
        for rid, st in cur.items():
            prev = _last_run_status.get(rid)
            if prev is None:
                # First time we see it; record but only log if we've ever
                # observed the runs page before.
                if _last_run_status:
                    _log_event(f"NEW_RUN id={rid} status={st}")
            elif prev != st:
                _log_event(f"RUN_STATUS_CHANGE id={rid} {prev} -> {st}")
        _last_run_status.update(cur)


# ---------- Main loop ----------
def _one_pass(ws):
    for slug, url in URLS:
        try:
            _navigate_and_wait(ws, url)
            shot = _capture_screenshot(ws, slug)
            tpath, text = _capture_text(ws, slug)
            print(f"[{_now_iso()}] captured {slug} screenshot={os.path.basename(shot) if shot else 'NONE'} "
                  f"text={os.path.basename(tpath)} text_len={len(text)}", flush=True)
            _process_changes(slug, text)
        except Exception as e:
            _log_event(f"ERROR capturing {slug}: {e}")
            # Re-raise to trigger outer reconnect on hard WS errors
            if isinstance(e, (websocket.WebSocketException, ConnectionError, OSError)):
                raise


def main():
    _ensure_dirs()
    _log_event("dom_screenshot_daemon starting")

    while True:
        try:
            target = _wait_for_cdp()
            ws = _connect_ws(target)
            _log_event("WS connected to CDP target")
            try:
                while True:
                    start = time.time()
                    _one_pass(ws)
                    elapsed = time.time() - start
                    sleep_s = max(1.0, POLL_INTERVAL_S - elapsed)
                    time.sleep(sleep_s)
            finally:
                try:
                    ws.close()
                except Exception:
                    pass
        except KeyboardInterrupt:
            _log_event("KeyboardInterrupt — exiting")
            return
        except Exception as e:
            _log_event(f"FATAL loop error: {e}\n{traceback.format_exc()}")
            time.sleep(ERROR_RETRY_S)


if __name__ == "__main__":
    main()
