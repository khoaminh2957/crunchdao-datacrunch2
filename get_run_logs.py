"""Fetch detailed logs + error for run #81535."""
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

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "81535"
url = f"https://hub.crunchdao.com/competitions/datacrunch-2/models/12867/curly-crayfishwetminh-v2/runs/{RUN_ID}"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(8)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if f"runs/{RUN_ID}" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

for i in range(20):
    time.sleep(1)
    n = evaluate(ws, "document.body.innerText.length", 1+i)
    if isinstance(n, int) and n > 4000: break

# Click "Logs" tab if present
evaluate(ws, r"""
  (() => {
    const link = Array.from(document.querySelectorAll('button, a')).find(b => /^(Logs?|Output|Console)$/i.test(b.innerText.trim()));
    if (link) link.click();
  })()
""", 50)
time.sleep(3)

state = evaluate(ws, "({text: document.body.innerText.slice(0, 12000)})", 100)
print(f"=== RUN {RUN_ID} ===")
print(state.get('text',''))
ws.close()
