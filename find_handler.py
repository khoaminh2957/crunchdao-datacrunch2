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

r = evaluate(ws, r"""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
    if (!btn) return {error: 'no btn'};
    // Walk up DOM tree to find form OR nearest meaningful parent
    let p = btn.parentElement;
    const path = [];
    let formFound = null;
    for (let i = 0; i < 15 && p; i++) {
      path.push({tag: p.tagName, classes: p.className.slice(0,80), role: p.getAttribute('role')});
      if (p.tagName === 'FORM') formFound = i;
      p = p.parentElement;
    }
    return {
      btn_type: btn.type,
      btn_form: btn.form ? 'has form attr' : 'no form attr',
      btn_form_id: btn.form?.id,
      path: path.slice(0, 12),
      form_found_at_depth: formFound,
      // Get React fiber via internal property
      has_react_fiber: !!(Object.keys(btn).find(k => k.startsWith('__reactProps') || k.startsWith('__reactFiber')))
    };
  })()
""", 1)
print(json.dumps(r, indent=2, ensure_ascii=False)[:3000])

# Inspect React props onClick handler
r2 = evaluate(ws, r"""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
    const propsKey = Object.keys(btn).find(k => k.startsWith('__reactProps$'));
    if (!propsKey) return {error: 'no react props'};
    const props = btn[propsKey];
    return {
      keys: Object.keys(props),
      onClick_type: typeof props.onClick,
      disabled: props.disabled,
      type: props.type
    };
  })()
""", 2)
print("\nREACT PROPS:", r2)

# Call React onClick directly if exists
if r2 and r2.get('onClick_type') == 'function':
    print("\nCalling React onClick directly...")
    r3 = evaluate(ws, r"""
      (() => {
        const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
        const propsKey = Object.keys(btn).find(k => k.startsWith('__reactProps$'));
        const props = btn[propsKey];
        try {
          // Create synthetic React event
          const fakeEvent = {
            preventDefault: () => {}, stopPropagation: () => {},
            target: btn, currentTarget: btn, type: 'click', bubbles: true
          };
          props.onClick(fakeEvent);
          return {called: true};
        } catch(e) { return {error: String(e).slice(0,200)}; }
      })()
    """, 3)
    print("CALL:", r3)
    time.sleep(8)
    state = evaluate(ws, "({url: location.href})", 4)
    print("URL:", state)
ws.close()
