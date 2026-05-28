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
    if "exceptionDetails" in r.get("result", {}):
        print("JS_ERR:", json.dumps(r["result"]["exceptionDetails"])[:200])
    return r.get("result", {}).get("result", {}).get("value")

# Find form's React fiber and walk up to find useState/useForm hooks
r = evaluate(ws, r"""
  (() => {
    const form = document.querySelector('form');
    const fiberKey = Object.keys(form).find(k => k.startsWith('__reactFiber$'));
    let fiber = form[fiberKey];
    // Walk fiber tree up to find component with state
    const states = [];
    let depth = 0;
    while (fiber && depth < 20) {
      if (fiber.memoizedState !== null && typeof fiber.memoizedState === 'object') {
        try {
          let s = fiber.memoizedState;
          const stateChain = [];
          let i = 0;
          while (s && i < 15) {
            stateChain.push({type: typeof s.memoizedState, val: typeof s.memoizedState === 'object' ? JSON.stringify(s.memoizedState).slice(0,200) : String(s.memoizedState).slice(0,100)});
            s = s.next;
            i++;
          }
          if (stateChain.length > 0) {
            states.push({depth, type: fiber.type?.name || fiber.elementType?.name || fiber.elementType || '?', stateChain});
          }
        } catch(e) {}
      }
      fiber = fiber.return;
      depth++;
    }
    return states;
  })()
""", 1)
print("REACT STATES:")
print(json.dumps(r, indent=2, ensure_ascii=False)[:4000])
ws.close()
