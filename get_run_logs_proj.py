"""Fetch detailed logs + error for a run. arg1=run_id, arg2=project_slug."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

def cdp(ws, method, params, mid):
    ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def evaluate(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True}, mid)
    return r.get("result", {}).get("result", {}).get("value")

RUN_ID = sys.argv[1]
PROJECT = sys.argv[2]
url = f"https://hub.crunchdao.com/competitions/datacrunch-2/models/12867/{PROJECT}/runs/{RUN_ID}"

# Close existing tab first
tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
for t in tabs:
    if f"runs/{RUN_ID}" in t.get("url",""):
        try: urllib.request.urlopen(f"http://localhost:9222/json/close/{t['id']}")
        except: pass
time.sleep(1)

req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(8)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if f"runs/{RUN_ID}" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

for i in range(20):
    time.sleep(1)
    n = evaluate(ws, "document.body.innerText.length", 1+i)
    if isinstance(n, int) and n > 3000: break

state = evaluate(ws, "({text: document.body.innerText.slice(0, 12000)})", 100)
print(f"=== RUN {RUN_ID} / {PROJECT} ===")
print(state.get('text',''))
ws.close()
