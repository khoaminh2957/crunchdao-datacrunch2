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

# Inspect "Force first train" section
r = evaluate(ws, r"""
  (() => {
    // Look for sections with "Force first train"
    const allText = document.body.innerText;
    const idx = allText.indexOf('Force first train');
    if (idx < 0) return {error: 'no_force_section'};
    // Find all radios in form
    const radios = Array.from(document.querySelectorAll('input[type="radio"]'));
    return {
      radio_count: radios.length,
      radios: radios.map(r => ({
        name: r.name, value: r.value, checked: r.checked,
        label_text: (r.closest('label')?.innerText || r.parentElement?.innerText || '').slice(0,50)
      }))
    };
  })()
""", 1)
print("RADIOS:", json.dumps(r, indent=2, ensure_ascii=False)[:3000])

# Click any unchecked radio that says "No" near "Force first train"
fix = evaluate(ws, r"""
  (() => {
    const radios = Array.from(document.querySelectorAll('input[type="radio"]'));
    // Pick last 2 unchecked (Force first train: No/Yes)
    const unchecked = radios.filter(r => !r.checked);
    const out = [];
    unchecked.forEach(r => {
      const proto = window.HTMLInputElement.prototype;
      const setter = Object.getOwnPropertyDescriptor(proto, 'checked').set;
      setter.call(r, true);
      r.dispatchEvent(new Event('change', {bubbles: true}));
      r.dispatchEvent(new Event('input', {bubbles: true}));
      r.click();
      out.push({val: r.value, name: r.name});
    });
    return out;
  })()
""", 2)
print("CLICKED:", fix)
time.sleep(2)

# Now try Create run again
final = evaluate(ws, r"""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
    return {disabled: btn?.disabled, clicked: btn && !btn.disabled ? (btn.click(), true) : false};
  })()
""", 3)
print("CREATE RUN ATTEMPT:", final)
time.sleep(8)

state = evaluate(ws, "({url: location.href, head: document.body.innerText.slice(0,1500)})", 4)
print("\nAFTER:", json.dumps(state, indent=2, ensure_ascii=False)[:2000])
ws.close()
