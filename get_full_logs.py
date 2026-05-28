import json, urllib.request, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)
tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if "runs/81456" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, method, params, mid):
    ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def evaluate(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True}, mid)
    return r.get("result", {}).get("result", {}).get("value")

# Click "Show advanced logs" toggle if exists
evaluate(ws, r"""
  (() => {
    const all = Array.from(document.querySelectorAll('button, input[type="checkbox"], label'));
    const adv = all.find(e => /advanced log/i.test((e.innerText||e.textContent||'').trim()));
    if (adv) adv.click();
  })()
""", 1)
time.sleep(2)

# Get full text
txt = evaluate(ws, "document.body.innerText", 2) or ""
# Find Logs section
idx = txt.find("Logs")
if idx >= 0:
    logs = txt[idx:idx+6000]
    print(logs)
ws.close()
