"""Atomic: reset all fields then click Create run with real CDP mouse."""
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

def click_coords(x, y, mid):
    cdp(ws, "Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y}, mid)
    cdp(ws, "Input.dispatchMouseEvent", {"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1}, mid+1)
    cdp(ws, "Input.dispatchMouseEvent", {"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1}, mid+2)

# Refresh page first
cdp(ws, "Page.enable", {}, 0)
cdp(ws, "Page.reload", {}, 1)
time.sleep(7)

# Get coords of each interactive element + click in proper sequence
coords = evaluate(ws, r"""
  (() => {
    const out = {};
    const yes = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Yes I did');
    if (yes) { yes.scrollIntoView(); const r = yes.getBoundingClientRect(); out.yes = {x: Math.round(r.x+r.width/2), y: Math.round(r.y+r.height/2)}; }
    const powerful = document.querySelector('button[role="radio"][value="aws-cpu"]');
    if (powerful) { powerful.scrollIntoView(); const r = powerful.getBoundingClientRect(); out.powerful = {x: Math.round(r.x+r.width/2), y: Math.round(r.y+r.height/2)}; }
    const combo = document.querySelector('button[role="combobox"]');
    if (combo) { combo.scrollIntoView(); const r = combo.getBoundingClientRect(); out.combo = {x: Math.round(r.x+r.width/2), y: Math.round(r.y+r.height/2)}; }
    const create = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
    if (create) { create.scrollIntoView(); const r = create.getBoundingClientRect(); out.create = {x: Math.round(r.x+r.width/2), y: Math.round(r.y+r.height/2)}; }
    return out;
  })()
""", 5)
print("COORDS:", coords)

# Click sequence with REAL CDP mouse + scroll into view each time
if coords.get('yes'):
    print("Click Yes I did")
    click_coords(coords['yes']['x'], coords['yes']['y'], 10)
    time.sleep(1)

if coords.get('powerful'):
    print("Click Powerful radio")
    click_coords(coords['powerful']['x'], coords['powerful']['y'], 20)
    time.sleep(1)

# For combobox: open + select No
if coords.get('combo'):
    print("Open combobox")
    click_coords(coords['combo']['x'], coords['combo']['y'], 30)
    time.sleep(1.5)
    # Find No option that appeared
    no_coords = evaluate(ws, r"""
      (() => {
        const opt = Array.from(document.querySelectorAll('[role="option"]')).find(o => o.innerText.trim() === 'No');
        if (opt) { const r = opt.getBoundingClientRect(); return {x: Math.round(r.x+r.width/2), y: Math.round(r.y+r.height/2)}; }
        return null;
      })()
    """, 40)
    if no_coords:
        print("Click No option")
        click_coords(no_coords['x'], no_coords['y'], 50)
        time.sleep(1)

# Refresh create coords after all interactions
final_coords = evaluate(ws, r"""
  (() => {
    const create = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
    if (!create) return null;
    create.scrollIntoView({block: 'center'});
    const r = create.getBoundingClientRect();
    return {x: Math.round(r.x+r.width/2), y: Math.round(r.y+r.height/2), disabled: create.disabled};
  })()
""", 60)
print("Final Create coords:", final_coords)

if final_coords and not final_coords.get('disabled'):
    print("Click Create run via REAL mouse")
    click_coords(final_coords['x'], final_coords['y'], 70)
    time.sleep(10)

state = evaluate(ws, "({url: location.href, head: document.body.innerText.slice(0,800)})", 80)
print("\nFINAL URL:", state.get('url') if state else None)
ws.close()
