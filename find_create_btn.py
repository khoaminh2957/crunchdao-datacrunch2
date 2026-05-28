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
    return r.get("result", {}).get("result", {}).get("value")

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
run_tab = [t for t in tabs if "runs/create" in t.get("url","") and "crunchdao" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(run_tab["webSocketDebuggerUrl"], **WS_KW)

# Find all buttons with detail
r = evaluate(ws, r"""
  (() => {
    const all = Array.from(document.querySelectorAll('button, input[type="submit"], a'));
    return all.filter(b => /create|submit|run|launch/i.test((b.innerText||b.value||'').trim())).map(b => ({
      tag: b.tagName, text: b.innerText.slice(0,40), value: b.value, type: b.type, disabled: b.disabled, classes: b.className.slice(0,80)
    }));
  })()
""", 1)
print(json.dumps(r, indent=2, ensure_ascii=False))

# Force click ANY button with "Create"
click = evaluate(ws, r"""
  (() => {
    // Find form submit button at bottom of form
    const buttons = Array.from(document.querySelectorAll('button'));
    // Filter by exact text match
    let target = buttons.find(b => b.innerText.trim() === 'Create run');
    if (!target) {
      // Try by type=submit
      target = buttons.find(b => b.type === 'submit' && !b.disabled);
    }
    if (target) {
      target.scrollIntoView();
      target.click();
      return {clicked: true, text: target.innerText.slice(0,40), type: target.type};
    }
    return {clicked: false};
  })()
""", 2)
print("CLICK:", click)
time.sleep(8)

# Check redirect
state = evaluate(ws, "({url: location.href, text_head: document.body.innerText.slice(0,1500)})", 3)
print("\nAFTER:", json.dumps(state, indent=2, ensure_ascii=False)[:2000])
ws.close()
