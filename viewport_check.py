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

# Check viewport + force scroll into view + get fresh coords
r = evaluate(ws, r"""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
    if (!btn) return {error: 'no_btn'};
    btn.scrollIntoView({block: 'center', inline: 'center'});
    // Force window scroll explicitly
    const rect = btn.getBoundingClientRect();
    return {
      vw: window.innerWidth,
      vh: window.innerHeight,
      btn_x: rect.x,
      btn_y: rect.y,
      btn_w: rect.width,
      btn_h: rect.height,
      page_y_offset: window.pageYOffset,
      doc_height: document.documentElement.scrollHeight,
      disabled: btn.disabled,
      visible_in_viewport: rect.x >= 0 && rect.y >= 0 && rect.x + rect.width <= window.innerWidth && rect.y + rect.height <= window.innerHeight
    };
  })()
""", 1)
print("VIEWPORT:", r)
ws.close()
