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
        print("JS_ERR:", json.dumps(r["result"]["exceptionDetails"])[:200])
    return r.get("result", {}).get("result", {}).get("value")

# Click combobox + select No
r = evaluate(ws, r"""
  (() => {
    // Open the Force first train combobox
    const combo = document.querySelector('button[role="combobox"]');
    if (!combo) return {error: 'no_combo'};
    combo.click();
    return {opened: true};
  })()
""", 1)
print("OPEN:", r)
time.sleep(2)

# Now select "No" option that appears
r2 = evaluate(ws, r"""
  (() => {
    // Find listbox option "No"
    const options = Array.from(document.querySelectorAll('[role="option"], [data-radix-collection-item]'));
    const noOpt = options.find(o => o.innerText.trim() === 'No');
    if (noOpt) { noOpt.click(); return {clicked: 'No'}; }
    return {clicked: null, options: options.map(o => o.innerText.slice(0,20))};
  })()
""", 2)
print("SELECT NO:", r2)
time.sleep(2)

# Also try setting number input train freq via real keystroke
r3 = evaluate(ws, r"""
  (() => {
    const num = document.querySelector('input[type="number"]');
    if (num) {
      num.focus();
      // Simulate user typing
      const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
      setter.call(num, '0');
      num.dispatchEvent(new Event('input', {bubbles: true}));
      num.dispatchEvent(new Event('change', {bubbles: true}));
      num.blur();
      return {set: '0', val: num.value};
    }
    return {error: 'no num'};
  })()
""", 3)
print("NUM:", r3)
time.sleep(1)

# Now click Create run
r4 = evaluate(ws, r"""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
    if (!btn) return {error: 'no btn'};
    const before = btn.disabled;
    btn.scrollIntoView({block: 'center'});
    btn.click();
    return {clicked: true, was_disabled: before, classes: btn.className.includes('disabled') ? 'has-disabled-class' : 'no-disabled-class'};
  })()
""", 4)
print("CREATE:", r4)
time.sleep(10)

state = evaluate(ws, "({url: location.href, head: document.body.innerText.slice(0,1500)})", 5)
print("\nAFTER:", json.dumps(state, indent=2, ensure_ascii=False)[:2000])
ws.close()
