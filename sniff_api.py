"""Wrap fetch to log all calls, then trigger click."""
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

# Wrap fetch
evaluate(ws, r"""
  window.__fetchLog = [];
  const origFetch = window.fetch;
  window.fetch = function(url, opts) {
    window.__fetchLog.push({url: typeof url === 'string' ? url : url.url, method: opts?.method || 'GET', body: opts?.body});
    return origFetch.apply(this, arguments);
  };
""", 1)
print("Fetch wrapped")
time.sleep(0.5)

# Click button - try mouseDown+Up + click event manually
r = evaluate(ws, r"""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
    if (!btn) return {error: 'no_btn'};
    // Comprehensive click sequence
    btn.focus();
    const rect = btn.getBoundingClientRect();
    const ev_init = {bubbles: true, cancelable: true, view: window, clientX: rect.x + rect.width/2, clientY: rect.y + rect.height/2, button: 0};
    btn.dispatchEvent(new MouseEvent('mousedown', ev_init));
    btn.dispatchEvent(new MouseEvent('mouseup', ev_init));
    btn.dispatchEvent(new MouseEvent('click', ev_init));
    btn.click();  // also try native
    return {dispatched: true, btn_disabled: btn.disabled};
  })()
""", 2)
print("CLICK:", r)
time.sleep(5)

# Check fetch log
log = evaluate(ws, "JSON.stringify(window.__fetchLog || [])", 3)
print("FETCH CALLS:", log[:3000] if log else "empty")

# Final state
state = evaluate(ws, "({url: location.href, head: document.body.innerText.slice(0,1000)})", 4)
print("\nURL:", state.get('url'))
ws.close()
