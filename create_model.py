"""Click 'Create Model' on CrunchDAO hub, fill form, submit."""
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
cd = [t for t in tabs if "hub.crunchdao.com" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(cd["webSocketDebuggerUrl"], **WS_KW)

# Click Create Model button
click = evaluate(ws, r"""
  (() => {
    const all = Array.from(document.querySelectorAll('button, a, [role="button"]'));
    const btn = all.find(e => /create model/i.test((e.innerText||'').trim()));
    if (btn) { btn.click(); return {clicked: true, text: btn.innerText, href: btn.href}; }
    return {clicked: false, available: all.map(e => e.innerText.slice(0,30)).filter(t=>t).slice(0,15)};
  })()
""", 1)
print("CLICK:", click)
time.sleep(6)

# Check for form
state = evaluate(ws, r"""
  (() => ({
    url: location.href,
    inputs: Array.from(document.querySelectorAll('input, select, textarea')).map(i => ({
      type: i.type, name: i.name, id: i.id, placeholder: i.placeholder,
      label: (i.labels && i.labels[0]) ? i.labels[0].innerText.slice(0,50) : null
    })).slice(0,15),
    buttons: Array.from(document.querySelectorAll('button')).map(b => b.innerText.slice(0,30)).filter(t=>t).slice(0,15),
    body_head: document.body.innerText.slice(0,1500)
  }))()
""", 2)
print("\nFORM STATE:", json.dumps(state, indent=2, ensure_ascii=False)[:3000])
ws.close()
