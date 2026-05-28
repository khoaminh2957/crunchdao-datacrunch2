"""DOM-based monitor for CrunchDAO datacrunch-2 runs (one poll, stdout JSON).

PowerShell wrapper (monitor_continuous.ps1) calls this every 60s.
Uses existing CDP browser on port 9222 (already authed to hub.crunchdao.com).

Output: single JSON line on stdout with:
  {
    "ts": "2026-05-26T12:34:56Z",
    "runs": [
       {"submission": 1, "run_id": "...", "status": "...", "duration": "...", "error": "..."},
       ...
    ],
    "ok": true|false,
    "fatal": "..."   # only if ok=false
  }
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

try:
    import websocket  # websocket-client
except Exception as e:
    print(json.dumps({"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "ok": False, "fatal": f"websocket import failed: {e}", "runs": []}))
    sys.exit(0)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

WS_KW = dict(origin="http://localhost", suppress_origin=False)

PROJECT_BASE = "https://hub.crunchdao.com/competitions/datacrunch-2/projects/12867/curly-crayfishwetminh-v2"
API_RUNS = ("https://api.hub.crunchdao.com/v3/competitions/datacrunch-2/"
            "projects/12867/curly-crayfishwetminh-v2/runs")
SUBMISSIONS = [1, 2, 3]


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cdp(ws, method, params, mid, timeout=20):
    ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ws.settimeout(max(0.1, deadline - time.time()))
            r = json.loads(ws.recv())
        except Exception:
            continue
        if r.get("id") == mid:
            return r
    return {}


def evaluate(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate",
            {"expression": expr, "returnByValue": True, "awaitPromise": True},
            mid)
    return r.get("result", {}).get("result", {}).get("value")


def find_or_open_tab(url_substr, full_url):
    """Find an existing tab whose URL contains url_substr; else open full_url."""
    try:
        tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json", timeout=5).read())
    except Exception as e:
        return None, f"cdp list failed: {e}"
    page_tabs = [t for t in tabs if t.get("type") == "page"]
    for t in page_tabs:
        if url_substr in t.get("url", ""):
            return t, None
    # open new tab
    try:
        req = urllib.request.Request(
            f"http://localhost:9222/json/new?{urllib.parse.quote(full_url)}",
            method="PUT")
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        return None, f"open tab failed: {e}"
    time.sleep(4)
    try:
        tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json", timeout=5).read())
    except Exception as e:
        return None, f"cdp list2 failed: {e}"
    for t in tabs:
        if t.get("type") == "page" and url_substr in t.get("url", ""):
            return t, None
    return None, "tab not found after open"


def fetch_runs_via_api():
    """Use authed browser to call the JSON API directly. Returns list of run dicts."""
    tab, err = find_or_open_tab("hub.crunchdao.com", PROJECT_BASE + "/submissions")
    if tab is None:
        return None, err
    try:
        ws = websocket.create_connection(tab["webSocketDebuggerUrl"], timeout=10, **WS_KW)
    except Exception as e:
        return None, f"ws connect failed: {e}"
    try:
        # Ask the page to fetch the API with credentials.
        expr = r"""
          (async () => {
            const urls = [
              'https://api.hub.crunchdao.com/v3/competitions/datacrunch-2/projects/12867/curly-crayfishwetminh-v2/runs?submissionNumber=1',
              'https://api.hub.crunchdao.com/v3/competitions/datacrunch-2/projects/12867/curly-crayfishwetminh-v2/runs?submissionNumber=2',
              'https://api.hub.crunchdao.com/v3/competitions/datacrunch-2/projects/12867/curly-crayfishwetminh-v2/runs?submissionNumber=3'
            ];
            const out = [];
            for (const u of urls) {
              try {
                const r = await fetch(u, {credentials: 'include', headers: {'Accept':'application/json'}});
                const t = await r.text();
                let body = t;
                try { body = JSON.parse(t); } catch (e) {}
                out.push({url: u, status: r.status, body: body});
              } catch (e) {
                out.push({url: u, error: String(e)});
              }
            }
            return out;
          })()
        """
        res = evaluate(ws, expr, 1)
    finally:
        try:
            ws.close()
        except Exception:
            pass
    return res, None


def parse_api_runs(raw):
    """Flatten the per-submission API responses into a list of run records.

    The API may return either a list of runs, or {content:[...]}, or an error object.
    We pull common fields conservatively.
    """
    flat = []
    if not raw:
        return flat
    for entry in raw:
        sub_num = None
        try:
            qs = entry.get("url", "").split("?", 1)[-1]
            for kv in qs.split("&"):
                if kv.startswith("submissionNumber="):
                    sub_num = int(kv.split("=", 1)[1])
        except Exception:
            pass
        status_http = entry.get("status")
        body = entry.get("body")
        if isinstance(body, str):
            # non-JSON response (HTML/redirect) — record as one synthetic run
            flat.append({
                "submission": sub_num,
                "run_id": "",
                "status": f"HTTP_{status_http}",
                "duration": "",
                "error": (body or "")[:200].replace("\n", " "),
            })
            continue
        runs = []
        if isinstance(body, list):
            runs = body
        elif isinstance(body, dict):
            for key in ("content", "items", "runs", "data"):
                v = body.get(key)
                if isinstance(v, list):
                    runs = v
                    break
            if not runs and any(k in body for k in ("id", "status", "state")):
                runs = [body]
            if not runs and ("message" in body or "error" in body):
                flat.append({
                    "submission": sub_num,
                    "run_id": "",
                    "status": f"API_ERR_{status_http}",
                    "duration": "",
                    "error": str(body.get("message") or body.get("error"))[:200],
                })
                continue
        for r in runs:
            if not isinstance(r, dict):
                continue
            rid = (r.get("id") or r.get("runId") or r.get("uuid")
                   or r.get("number") or "")
            status = (r.get("status") or r.get("state")
                      or r.get("phase") or "")
            duration = (r.get("duration") or r.get("runtime")
                        or r.get("executionDuration") or "")
            err = ""
            for k in ("error", "errorMessage", "failureReason", "message"):
                v = r.get(k)
                if v:
                    err = str(v)[:200]
                    break
            flat.append({
                "submission": sub_num,
                "run_id": str(rid),
                "status": str(status),
                "duration": str(duration),
                "error": err,
            })
        if not runs and not (isinstance(body, dict) and ("message" in body or "error" in body)):
            # empty list — no runs for this submission
            flat.append({
                "submission": sub_num,
                "run_id": "",
                "status": "NONE",
                "duration": "",
                "error": "",
            })
    return flat


def scrape_dom_fallback():
    """Fallback: open each submission page and scrape document.body.innerText."""
    out = []
    for sn in SUBMISSIONS:
        url = f"{PROJECT_BASE}/submissions/{sn}"
        substr = f"submissions/{sn}"
        tab, err = find_or_open_tab(substr, url)
        if tab is None:
            out.append({"submission": sn, "run_id": "", "status": "TAB_ERR",
                        "duration": "", "error": err or ""})
            continue
        try:
            ws = websocket.create_connection(tab["webSocketDebuggerUrl"],
                                             timeout=10, **WS_KW)
        except Exception as e:
            out.append({"submission": sn, "run_id": "", "status": "WS_ERR",
                        "duration": "", "error": str(e)[:200]})
            continue
        try:
            for i in range(10):
                time.sleep(1)
                n = evaluate(ws, "document.body.innerText.length", 200 + i)
                if isinstance(n, int) and n > 800:
                    break
            text = evaluate(ws, "document.body.innerText.slice(0,4000)", 300)
            text = text or ""
            status = ""
            for kw in ("FAILED", "ERROR", "SUCCESS", "SUCCEEDED",
                       "RUNNING", "PENDING", "QUEUED", "CANCELLED", "CANCELED"):
                if kw in text.upper():
                    status = kw
                    break
            err = ""
            up = text.upper()
            if "ERROR" in up or "FAILED" in up:
                idx = up.find("ERROR")
                if idx < 0:
                    idx = up.find("FAILED")
                err = text[max(0, idx):idx + 300].replace("\n", " ")
            out.append({"submission": sn, "run_id": "", "status": status or "UNKNOWN",
                        "duration": "", "error": err[:200]})
        finally:
            try:
                ws.close()
            except Exception:
                pass
    return out


def main():
    payload = {"ts": now_iso(), "ok": True, "runs": [], "source": "api"}
    raw, err = fetch_runs_via_api()
    if err:
        payload["ok"] = False
        payload["fatal"] = err
        # try DOM fallback
        try:
            payload["runs"] = scrape_dom_fallback()
            payload["source"] = "dom_fallback"
            payload["ok"] = bool(payload["runs"])
        except Exception as e2:
            payload["fatal"] = f"{err}; dom_fallback: {e2}"
    else:
        try:
            payload["runs"] = parse_api_runs(raw)
        except Exception as e:
            payload["ok"] = False
            payload["fatal"] = f"parse failed: {e}"
            try:
                payload["runs"] = scrape_dom_fallback()
                payload["source"] = "dom_fallback"
                payload["ok"] = bool(payload["runs"])
            except Exception as e2:
                payload["fatal"] += f"; dom_fallback: {e2}"
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
