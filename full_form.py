"""Click all required inputs on Create Run form."""
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
run_tab = [t for t in tabs if "runs/create" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(run_tab["webSocketDebuggerUrl"], **WS_KW)

# Probe ALL inputs + radios on the page
probe = evaluate(ws, r"""
  (() => {
    const radios = Array.from(document.querySelectorAll('input[type="radio"]'));
    const numInputs = Array.from(document.querySelectorAll('input[type="number"], input[type="text"]'));
    // Powerful card might be a div with role=button or label wrapping radio
    const cards = Array.from(document.querySelectorAll('label, [role="radio"], [data-state]'));
    return {
      radios: radios.map(r => ({name: r.name, value: r.value, checked: r.checked, id: r.id})),
      num_inputs: numInputs.map(i => ({name: i.name, value: i.value, id: i.id, placeholder: i.placeholder})),
      cards_first10: cards.slice(0,10).map(c => ({tag: c.tagName, text: c.innerText.slice(0,40), dataState: c.getAttribute('data-state'), role: c.getAttribute('role')}))
    };
  })()
""", 1)
print("PROBE:", json.dumps(probe, indent=2, ensure_ascii=False)[:3500])
ws.close()
