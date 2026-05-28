import json, urllib.request, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)
tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab_list = [t for t in tabs if "runs/81520" in t.get("url","") and t.get("type") == "page"]
if not tab_list:
    import urllib.parse
    req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote('https://hub.crunchdao.com/competitions/datacrunch-2/models/12867/curly-crayfishwetminh-v2/runs/81520')}", method="PUT")
    urllib.request.urlopen(req); time.sleep(8)
    tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
    tab_list = [t for t in tabs if "runs/81520" in t.get("url","") and t.get("type") == "page"]

ws = websocket.create_connection(tab_list[0]["webSocketDebuggerUrl"], **WS_KW)
def cdp(ws, method, params, mid):
    ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r
def evaluate(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True}, mid)
    return r.get("result", {}).get("result", {}).get("value")

cdp(ws, "Page.enable", {}, 0); cdp(ws, "Page.reload", {}, 1); time.sleep(8)
txt = evaluate(ws, "document.body.innerText", 2) or ""
import re
status = re.search(r"Status\s*\n\s*(\w+)", txt)
duration = re.search(r"Duration\s*\n\s*([^\n]+)", txt)
err = re.search(r"Error\s*\n\s*(.{20,200})", txt)
print(f"Status: {status.group(1) if status else '?'}")
print(f"Duration: {duration.group(1) if duration else '?'}")
print(f"Error: {err.group(1)[:150] if err else 'none'}")
# Get last 1000 chars (latest logs)
print(f"\n=== Last log section ===")
print(txt[-2500:])
ws.close()
