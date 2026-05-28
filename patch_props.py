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

# Wrap fetch to capture
evaluate(ws, "window.__fetchLog = []; const _f = window.fetch; window.fetch = function(u,o){const url = typeof u==='string'?u:u.url; window.__fetchLog.push({u:url,m:o?.method||'GET',b:o?.body?.toString().slice(0,500)}); return _f.apply(this,arguments);};", 0)

# Patch React props.disabled = false + invoke onClick
r = evaluate(ws, r"""
  (async () => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
    const propsKey = Object.keys(btn).find(k => k.startsWith('__reactProps$'));
    const props = btn[propsKey];

    // Force disabled to false in props
    props.disabled = false;
    btn.disabled = false;
    btn.removeAttribute('disabled');

    // Now try fire onClick via the actual React event system
    // Create proper synthetic event
    const fakeEvt = new MouseEvent('click', {bubbles: true, cancelable: true});
    // Add React-expected properties
    Object.defineProperty(fakeEvt, 'isDefaultPrevented', {value: () => false});
    Object.defineProperty(fakeEvt, 'isPropagationStopped', {value: () => false});

    try {
      await props.onClick(fakeEvt);
      return {invoked: true};
    } catch(e) {
      return {error: String(e).slice(0,300)};
    }
  })()
""", 1)
print("INVOKE:", r)
time.sleep(10)

# Check fetch log
log = evaluate(ws, "JSON.stringify(window.__fetchLog || [])", 2)
print("\nFETCH LOG:")
if log:
    print(log[:3000])

state = evaluate(ws, "({url: location.href, head: document.body.innerText.slice(0,800)})", 3)
print("\nURL:", state.get('url') if state else None)
ws.close()
