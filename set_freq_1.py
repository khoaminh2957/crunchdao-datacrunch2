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

# Set train_frequency = 1 via real keyboard
import time as t
num_focused = evaluate(ws, "(() => { const n = document.querySelector('input[type=\"number\"]'); n.focus(); n.select(); return {ok: true}; })()", 1)
print("Focus num:", num_focused)
# Clear via backspace
cdp(ws, "Input.dispatchKeyEvent", {"type": "keyDown", "key": "Backspace"}, 2)
cdp(ws, "Input.dispatchKeyEvent", {"type": "keyUp", "key": "Backspace"}, 3)
t.sleep(0.3)
# Type 1
cdp(ws, "Input.insertText", {"text": "1"}, 4)
t.sleep(0.3)
# Tab off to commit
cdp(ws, "Input.dispatchKeyEvent", {"type": "keyDown", "key": "Tab"}, 5)
cdp(ws, "Input.dispatchKeyEvent", {"type": "keyUp", "key": "Tab"}, 6)
t.sleep(1)

# Check actual button disabled state now
r = evaluate(ws, r"""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
    return {disabled: btn.disabled, aria_disabled: btn.getAttribute('aria-disabled'), num_val: document.querySelector('input[type="number"]').value};
  })()
""", 7)
print("STATE:", r)

# Now click via real CDP coords
if r and not r.get('disabled'):
    coords = evaluate(ws, r"""
      (() => {
        const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
        btn.scrollIntoView({block: 'center'});
        const rect = btn.getBoundingClientRect();
        return {x: Math.round(rect.x + rect.width/2), y: Math.round(rect.y + rect.height/2)};
      })()
    """, 8)
    print("Coords:", coords)
    if coords:
        x, y = coords['x'], coords['y']
        cdp(ws, "Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y}, 10)
        cdp(ws, "Input.dispatchMouseEvent", {"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1}, 11)
        cdp(ws, "Input.dispatchMouseEvent", {"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1}, 12)
        t.sleep(8)

state = evaluate(ws, "({url: location.href, head: document.body.innerText.slice(0,1500)})", 20)
print("\nAFTER:", json.dumps(state, indent=2, ensure_ascii=False)[:2000])
ws.close()
