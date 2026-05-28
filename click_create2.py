import json, urllib.request, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)
tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
run_tab = [t for t in tabs if "runs/create" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(run_tab["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, method, params, mid):
    ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def evaluate(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True}, mid)
    return r.get("result", {}).get("result", {}).get("value")

# Check current state of Create run button + try multiple methods
r = evaluate(ws, r"""
  (() => {
    const all = Array.from(document.querySelectorAll('button'));
    const create = all.find(b => b.innerText.trim() === 'Create run');
    if (!create) return {error: 'not_found'};
    const state = {disabled: create.disabled, type: create.type, classes: create.className.slice(0,100)};
    // Try methods in order
    if (!create.disabled) {
      create.click();
      return {clicked_normal: true, state};
    }
    // Force-enable + click
    create.disabled = false;
    create.click();
    return {forced: true, state};
  })()
""", 1)
print("CREATE RUN:", r)
time.sleep(8)

state = evaluate(ws, "({url: location.href, text: document.body.innerText.slice(0,1500)})", 2)
print("\nAFTER:", json.dumps(state, indent=2, ensure_ascii=False)[:2000])
ws.close()
