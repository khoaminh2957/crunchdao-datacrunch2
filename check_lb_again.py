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

# Run details + score field via API
tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if "runs/81520" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

# Direct API
r = evaluate(ws, r"""
  (async () => {
    const res = await fetch('https://api.hub.crunchdao.com/v3/competitions/datacrunch-2/projects/12867/curly-crayfishwetminh-v2/runs/81520', {credentials: 'include'});
    return await res.json();
  })()
""", 1)
print("RUN API:", json.dumps(r, indent=2, ensure_ascii=False)[:3000])

# Submissions list
r2 = evaluate(ws, r"""
  (async () => {
    const res = await fetch('https://api.hub.crunchdao.com/v3/competitions/datacrunch-2/projects/12867/curly-crayfishwetminh-v2/runs?submissionNumber=2', {credentials: 'include'});
    return await res.json();
  })()
""", 2)
print("\nRUNS LIST:", json.dumps(r2, indent=2, ensure_ascii=False)[:2000])
ws.close()
