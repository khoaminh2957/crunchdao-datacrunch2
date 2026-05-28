"""Configure + submit the cloud run."""
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
run_tab = [t for t in tabs if "runs/create" in t.get("url","") and "crunchdao" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(run_tab["webSocketDebuggerUrl"], **WS_KW)

# Click "Yes I did" + ensure Powerful instance + click Create run
result = evaluate(ws, r"""
  (() => {
    const out = {};
    const all = Array.from(document.querySelectorAll('button, a, [role="button"], input[type="radio"], label'));

    // 1. Click "Yes I did"
    const yes = all.find(b => /^Yes I did$/i.test((b.innerText||'').trim()));
    if (yes) { yes.click(); out.yes_clicked = true; }

    // 2. Click "Powerful" instance radio (likely first option / already default)
    const powerful = all.find(b => /^Powerful$/.test((b.innerText||'').trim()));
    if (powerful) { powerful.click(); out.powerful_clicked = true; }

    return out;
  })()
""", 1)
print("CONFIG:", result)
time.sleep(2)

# Now click "Create run"
create = evaluate(ws, r"""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => /^Create run$/i.test((b.innerText||'').trim()) && !b.disabled);
    if (btn) { btn.click(); return {clicked: true}; }
    return {clicked: false, available: Array.from(document.querySelectorAll('button')).map(b => `${b.innerText.slice(0,30)}(disabled=${b.disabled})`).slice(0,15)};
  })()
""", 2)
print("CREATE RUN:", create)
time.sleep(8)

# Verify
state = evaluate(ws, r"""
  (() => ({
    url: location.href,
    text: document.body.innerText.slice(0,2000)
  }))()
""", 3)
print("\nAFTER:", json.dumps(state, indent=2, ensure_ascii=False)[:2500])
ws.close()
