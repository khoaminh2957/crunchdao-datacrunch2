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

# Wrap fetch + XHR properly with simple script
wrap_script = """
window.__fetchLog = [];
const origFetch = window.fetch;
window.fetch = function() {
  const url = typeof arguments[0] === 'string' ? arguments[0] : arguments[0].url;
  const opts = arguments[1] || {};
  window.__fetchLog.push({url: url, method: opts.method || 'GET', body: String(opts.body || '').slice(0,500)});
  return origFetch.apply(this, arguments);
};
window.__xhrLog = [];
const OrigXHR = window.XMLHttpRequest;
function PatchedXHR() {
  const xhr = new OrigXHR();
  const origOpen = xhr.open;
  xhr.open = function(method, url) {
    window.__xhrLog.push({method: method, url: url});
    return origOpen.apply(this, arguments);
  };
  return xhr;
}
window.XMLHttpRequest = PatchedXHR;
"""
evaluate(ws, wrap_script, 0)
print("Wrappers installed")

# Now click via React props (force disabled false)
r = evaluate(ws, r"""
  (async () => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
    const propsKey = Object.keys(btn).find(k => k.startsWith('__reactProps$'));
    const props = btn[propsKey];
    btn.disabled = false;
    btn.removeAttribute('disabled');
    if (props) {
      props.disabled = false;
      try {
        const fakeEvt = new MouseEvent('click', {bubbles: true, cancelable: true});
        await props.onClick(fakeEvt);
        return {clicked: true};
      } catch(e) { return {error: String(e).slice(0,200)}; }
    }
    return {no_props: true};
  })()
""", 1)
print("CLICK:", r)
time.sleep(8)

# Check both logs
fetch_log = evaluate(ws, "JSON.stringify(window.__fetchLog || [])", 2)
xhr_log = evaluate(ws, "JSON.stringify(window.__xhrLog || [])", 3)
print("\nFETCH:", fetch_log[:2000] if fetch_log else "empty")
print("\nXHR:", xhr_log[:2000] if xhr_log else "empty")

# Check URL
state = evaluate(ws, "({url: location.href})", 4)
print("\nURL:", state)
ws.close()
