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

# Detailed form inspection
r = evaluate(ws, r"""
  (() => {
    const form = document.querySelector('form');
    const radios = Array.from(document.querySelectorAll('input[type="radio"], button[role="radio"], [data-state]'));
    const selects = Array.from(document.querySelectorAll('select, [role="combobox"]'));
    return {
      form_html: form?.outerHTML.slice(0, 3000),
      radios_count: radios.length,
      selects: selects.map(s => ({tag: s.tagName, role: s.getAttribute('role'), text: s.innerText.slice(0,80), data_state: s.getAttribute('data-state')}))
    };
  })()
""", 1)
print("FORM:", json.dumps(r, indent=2, ensure_ascii=False)[:5000])
ws.close()
