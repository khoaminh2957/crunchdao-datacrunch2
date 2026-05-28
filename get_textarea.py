import json, urllib.request, websocket, time, sys, re
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)
tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
cd = [t for t in tabs if "hub.crunchdao.com" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(cd["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, method, params, mid):
    ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def evaluate(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate",
            {"expression": expr, "returnByValue": True, "awaitPromise": True}, mid)
    return r.get("result", {}).get("result", {}).get("value")

# Get ALL textarea values fully
r = evaluate(ws, r"""
  (() => {
    const tas = Array.from(document.querySelectorAll('textarea'));
    return tas.map(t => ({val: t.value, len: t.value.length, id: t.id, className: t.className.slice(0,80)}));
  })()
""", 1)
print(json.dumps(r, indent=2, ensure_ascii=False)[:5000])

# Find crunch setup in any textarea
for ta in r or []:
    val = ta.get('val','')
    m = re.search(r'crunch setup \S+ \S+ --token \S+', val)
    if m:
        cmd = m.group(0)
        print(f"\n✓ FOUND: {cmd}")
        with open(r"C:\Users\Admin\earn5usd\crunchdao\fresh_setup.txt", "w") as f:
            f.write(cmd)
        break
ws.close()
