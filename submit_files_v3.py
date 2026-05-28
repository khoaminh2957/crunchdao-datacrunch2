"""Find HIDDEN file inputs (React dropzone pattern)."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

MAIN_PY = r"C:\Users\Admin\earn5usd\crunchdao\v60_submission_files\main.py"
REQ_TXT = r"C:\Users\Admin\earn5usd\crunchdao\v60_submission_files\requirements.txt"

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if "submit/files" in t.get("url","") and "curly" in t.get("url","") and t.get("type") == "page"][-1]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

def cdp(m, p, mid):
    ws.send(json.dumps({"id":mid,"method":m,"params":p or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def ev(expr, mid):
    r = cdp("Runtime.evaluate", {"expression":expr,"returnByValue":True,"awaitPromise":True}, mid)
    if "exceptionDetails" in r.get("result",{}):
        print("JS_ERR:", json.dumps(r["result"]["exceptionDetails"])[:300])
    return r.get("result",{}).get("result",{}).get("value")

cdp("Network.enable", {}, 1)
cdp("DOM.enable", {}, 2)

# Search all inputs even hidden
inputs = ev("""
  (() => {
    return Array.from(document.querySelectorAll('input')).map((el, i) => ({
      i, type: el.type, accept: el.accept, multiple: el.multiple,
      hidden: el.hidden, display: getComputedStyle(el).display,
      opacity: getComputedStyle(el).opacity, visibility: getComputedStyle(el).visibility,
      name: el.name, id: el.id,
    }));
  })()
""", 10)
print("All inputs:")
for i in inputs or []:
    print(f"  {i}")

# Search for dropzones / clickable upload areas
zones = ev("""
  (() => {
    // Common dropzone class names + clickable Files boxes
    const candidates = Array.from(document.querySelectorAll('[class*="dropzone"], [class*="file-upload"], [class*="upload"], button, div'))
      .filter(el => /Choose File|Upload|Drop file/i.test(el.innerText))
      .filter(el => el.offsetParent !== null);
    return candidates.slice(0, 5).map((el, i) => ({
      i, tag: el.tagName, text: el.innerText.trim().slice(0, 60), class: el.className.slice(0, 80),
    }));
  })()
""", 11)
print("\nClickable upload zones:")
for z in zones or []:
    print(f"  {z}")

# Try clicking "Choose File(s)" area to reveal hidden input
ev("""
  (() => {
    const cands = Array.from(document.querySelectorAll('button, div, label')).filter(e => /Choose File/i.test(e.innerText) && e.offsetParent !== null);
    if (cands.length) cands[0].click();
  })()
""", 20)
time.sleep(2)

# Re-search inputs
inputs2 = ev("""
  (() => {
    return Array.from(document.querySelectorAll('input[type="file"]')).map((el, i) => ({
      i, type: el.type, accept: el.accept, multiple: el.multiple, hidden: el.hidden,
    }));
  })()
""", 30)
print("\nFile inputs after clicking:")
print(inputs2)

# Use DOM API to find file input via attribute selector even if hidden
doc = cdp("DOM.getDocument", {"depth": -1}, 40)
root_id = doc["result"]["root"]["nodeId"]

# Try querySelectorAll on entire document tree
result = cdp("DOM.querySelectorAll", {"nodeId": root_id, "selector": 'input[type="file"]'}, 41)
print(f"\nDOM.querySelectorAll input[type='file']: {result['result']['nodeIds']}")

result2 = cdp("DOM.querySelectorAll", {"nodeId": root_id, "selector": 'input'}, 42)
print(f"DOM.querySelectorAll input (any): {len(result2['result']['nodeIds'])} found")

# Print attrs of each input
for nid in result2["result"]["nodeIds"][:20]:
    desc = cdp("DOM.describeNode", {"nodeId": nid}, 100+nid)
    n = desc["result"]["node"]
    attrs = dict(zip(n["attributes"][::2], n["attributes"][1::2])) if "attributes" in n else {}
    print(f"  nodeId={nid} type={attrs.get('type','?')} hidden={attrs.get('hidden','no')} class={attrs.get('class','')[:50]}")

ws.close()
