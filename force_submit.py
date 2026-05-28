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

# Use form.requestSubmit() - triggers React form properly
r = evaluate(ws, r"""
  (() => {
    const form = document.querySelector('form');
    if (!form) return {error: 'no_form'};
    const btn = Array.from(form.querySelectorAll('button[type="submit"]')).find(b => b.innerText.trim() === 'Create run');
    if (!btn) return {error: 'no_submit_btn'};
    // requestSubmit is the proper way to submit React forms
    form.requestSubmit(btn);
    return {submitted: true};
  })()
""", 1)
print("REQUEST_SUBMIT:", r)
time.sleep(10)

state = evaluate(ws, "({url: location.href, text_head: document.body.innerText.slice(0,2000)})", 2)
print("\nAFTER:", json.dumps(state, indent=2, ensure_ascii=False)[:2500])
ws.close()
