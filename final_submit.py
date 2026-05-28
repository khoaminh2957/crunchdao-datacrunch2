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

# Re-select Powerful (in case my earlier clicks broke it) + click Create run via form submit
r = evaluate(ws, r"""
  (() => {
    // 1. Re-click Powerful radio (aws-cpu)
    const powerful = Array.from(document.querySelectorAll('button[role="radio"][value="aws-cpu"]'))[0];
    if (powerful) powerful.click();

    // 2. Make sure "Yes I did" is selected
    const yesBtn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Yes I did');
    if (yesBtn) yesBtn.click();

    return {powerful_clicked: !!powerful, yes_clicked: !!yesBtn};
  })()
""", 1)
print("RESET:", r)
time.sleep(2)

# Submit form natively
r2 = evaluate(ws, r"""
  (() => {
    const form = document.querySelector('form');
    if (!form) return {error: 'no_form'};
    // Find Create run button + click natively
    const btn = Array.from(form.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
    if (!btn) return {error: 'no_btn'};
    const before_disabled = btn.disabled;
    btn.click();
    return {clicked: true, was_disabled: before_disabled};
  })()
""", 2)
print("SUBMIT:", r2)
time.sleep(10)

state = evaluate(ws, "({url: location.href, text: document.body.innerText.slice(0,2500)})", 3)
print("\nAFTER:", json.dumps(state, indent=2, ensure_ascii=False)[:3000])
ws.close()
