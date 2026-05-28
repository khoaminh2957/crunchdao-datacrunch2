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

# Get button coordinates + check exact state
r = evaluate(ws, r"""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
    if (!btn) return {error: 'not found'};
    btn.scrollIntoView({block: 'center'});
    const rect = btn.getBoundingClientRect();
    return {
      x: Math.round(rect.x + rect.width/2),
      y: Math.round(rect.y + rect.height/2),
      disabled: btn.disabled,
      aria_disabled: btn.getAttribute('aria-disabled'),
      type: btn.type,
      classes: btn.className.slice(0,100)
    };
  })()
""", 1)
print("BUTTON:", r)
time.sleep(1)

# Real OS-level click via CDP
if r and not r.get('error') and not r.get('disabled'):
    x, y = r['x'], r['y']
    cdp(ws, "Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y}, 10)
    cdp(ws, "Input.dispatchMouseEvent", {"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1}, 11)
    cdp(ws, "Input.dispatchMouseEvent", {"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1}, 12)
    print("Real click dispatched at", x, y)
    time.sleep(10)

state = evaluate(ws, "({url: location.href, text_head: document.body.innerText.slice(0,1500)})", 20)
print("\nAFTER:", json.dumps(state, indent=2, ensure_ascii=False)[:2000])
ws.close()
