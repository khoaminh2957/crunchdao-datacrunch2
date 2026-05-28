"""Click 'Run in the Cloud' button on submission page."""
import json, urllib.request, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

def cdp(ws, method, params, mid):
    ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid:
            return r

def evaluate(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate",
            {"expression": expr, "returnByValue": True, "awaitPromise": True}, mid)
    if "exceptionDetails" in r.get("result", {}):
        print("JS_ERR:", json.dumps(r["result"]["exceptionDetails"])[:200])
    return r.get("result", {}).get("result", {}).get("value")

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
sub_tab = [t for t in tabs if "submissions/1" in t.get("url","") and "crunchdao" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(sub_tab["webSocketDebuggerUrl"], **WS_KW)

# Click "Run in the Cloud"
click = evaluate(ws, r"""
  (() => {
    const all = Array.from(document.querySelectorAll('button, a, [role="button"]'));
    const btn = all.find(e => /Run in the Cloud/i.test((e.innerText||'').trim()));
    if (btn) { btn.click(); return {clicked: true, text: btn.innerText, href: btn.href}; }
    return {clicked: false, options: all.map(e => e.innerText.slice(0,40)).filter(t=>t).slice(0,15)};
  })()
""", 1)
print("CLICK:", click)
time.sleep(8)

# Check for modal/dropdown asking which round/phase to run
state = evaluate(ws, r"""
  (() => ({
    url: location.href,
    text: document.body.innerText.slice(0,3000),
    buttons: Array.from(document.querySelectorAll('button')).map(b => b.innerText.slice(0,40)).filter(t=>t).slice(0,20)
  }))()
""", 2)
print("\nSTATE AFTER CLICK:")
print(json.dumps(state, indent=2, ensure_ascii=False)[:4000])
ws.close()
