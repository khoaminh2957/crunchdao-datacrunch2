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
        print("JS_ERR:", json.dumps(r["result"]["exceptionDetails"])[:300])
    return r.get("result", {}).get("result", {}).get("value")

# Wrap fetch to capture
evaluate(ws, "window.__fetchLog = []; const _f = window.fetch; window.fetch = function(u,o){window.__fetchLog.push({u:typeof u==='string'?u:u.url,m:o?.method||'GET',b:o?.body?.toString().slice(0,300)}); return _f.apply(this,arguments);};", 0)

# Dispatch real native submit event on form
r = evaluate(ws, r"""
  (() => {
    const form = document.querySelector('form');
    if (!form) return {error: 'no_form'};
    // Try multiple approaches
    const results = {};
    // 1. requestSubmit with btn
    const submitBtns = Array.from(form.querySelectorAll('button[type="submit"]'));
    results.submit_btns_count = submitBtns.length;
    if (submitBtns.length > 0) {
      try {
        form.requestSubmit(submitBtns[0]);
        results.requestSubmit_called = true;
      } catch(e) { results.requestSubmit_err = String(e).slice(0,100); }
    }
    return results;
  })()
""", 1)
print("FORM SUBMIT:", r)
time.sleep(8)

# Check fetch log
log = evaluate(ws, "JSON.stringify(window.__fetchLog || [])", 2)
print("\nFETCH LOG (post submit):")
print(log[:3000] if log else "empty")

state = evaluate(ws, "({url: location.href})", 3)
print("\nURL:", state)
ws.close()
