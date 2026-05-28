"""Submit via Files Submission tab: select curly, upload main.py + requirements.txt."""
import json, urllib.request, urllib.parse, websocket, time, sys, os
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

MAIN_PY = r"C:\Users\Admin\earn5usd\crunchdao\v60_submission_files\main.py"
REQ_TXT = r"C:\Users\Admin\earn5usd\crunchdao\v60_submission_files\requirements.txt"
TARGET_PROJECT = "curly-crayfishwetminh-v2"

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
# Find existing files submission tab
matches = [t for t in tabs if "submit/files" in t.get("url","") and t.get("type") == "page"]
if not matches:
    # Open new files submission page
    url = f"https://hub.crunchdao.com/competitions/datacrunch-2/submit/files?projectName={TARGET_PROJECT}"
    req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
    urllib.request.urlopen(req); time.sleep(8)
    tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
    matches = [t for t in tabs if "submit/files" in t.get("url","") and t.get("type") == "page"]

tab = matches[-1]
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

# 1. Change model dropdown to curly
print("Step 1: Change Model dropdown to curly...")
# Click the model dropdown
r = ev("""
  (() => {
    // Find model selector (button/combobox with 'bewildered-chickadee' text)
    const sels = Array.from(document.querySelectorAll('button')).filter(b =>
      b.innerText.trim() === 'bewildered-chickadee' || b.getAttribute('role') === 'combobox'
    );
    if (sels.length === 0) return {error: 'no selector', buttons_seen: Array.from(document.querySelectorAll('button')).slice(0,10).map(b=>b.innerText.trim().slice(0,30))};
    sels[0].click();
    return {clicked: true, txt: sels[0].innerText.trim()};
  })()
""", 10)
print("Dropdown:", r)
time.sleep(2)

# Click curly option
r = ev("""
  (() => {
    const items = Array.from(document.querySelectorAll('[role="option"], button, div')).filter(e => e.offsetParent !== null);
    const curly = items.find(e => /curly-crayfishwetminh-v2/.test(e.innerText) && e.innerText.length < 50);
    if (!curly) return {error: 'no curly option', sample: items.slice(0,15).map(e=>e.innerText.trim().slice(0,30))};
    curly.click();
    return {clicked: 'curly'};
  })()
""", 11)
print("Select curly:", r)
time.sleep(3)

# 2. Set files (both main.py and requirements.txt)
print("\nStep 2: Upload files...")
doc = cdp("DOM.getDocument", {}, 20)
root_id = doc["result"]["root"]["nodeId"]
file_inputs = cdp("DOM.querySelectorAll", {"nodeId": root_id, "selector": 'input[type="file"]'}, 21)
node_ids = file_inputs["result"]["nodeIds"]
print(f"File input nodeIds: {node_ids}")

# Set both files
cdp("DOM.setFileInputFiles", {"files": [MAIN_PY, REQ_TXT], "nodeId": node_ids[0]}, 50)
time.sleep(2)

fcheck = ev("""
  (() => {
    const inp = document.querySelector('input[type="file"]');
    return {n_files: inp.files.length, names: Array.from(inp.files).map(f => f.name)};
  })()
""", 60)
print("Files set:", fcheck)

# Dispatch change event
ev("""
  (() => {
    const inp = document.querySelector('input[type="file"]');
    inp.dispatchEvent(new Event('change', {bubbles: true}));
    inp.dispatchEvent(new Event('input', {bubbles: true}));
  })()
""", 65)
time.sleep(2)

# 3. Check submit button + verify model is curly
state = ev("""
  (() => {
    const modelBtn = Array.from(document.querySelectorAll('button')).find(b =>
      /curly|bewildered|historic|efficient/.test(b.innerText) && b.innerText.length < 50
    );
    const subBtns = Array.from(document.querySelectorAll('button')).filter(b => /Submit to/.test(b.innerText) && b.offsetParent !== null);
    return {
      currentModel: modelBtn?.innerText.trim(),
      submitText: subBtns[0]?.innerText.trim(),
      submitDisabled: subBtns[0]?.disabled,
    };
  })()
""", 70)
print("Pre-submit state:", state)

# 4. Click submit
r = ev("""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => /Submit to/.test(b.innerText) && b.offsetParent !== null);
    if (!btn) return {error: 'no submit btn'};
    btn.click();
    return {clicked: true, text: btn.innerText.trim()};
  })()
""", 80)
print("Submit:", r)

# Capture network for 30s
events = []
ws.settimeout(0.5)
end_time = time.time() + 30
while time.time() < end_time:
    try:
        msg = json.loads(ws.recv())
        m = msg.get("method","")
        if m.startswith("Network.requestWillBeSent"):
            rd = msg["params"]["request"]
            u = rd.get("url","")
            if "api.hub.crunchdao.com" in u and rd.get("method") in ("POST","PUT","PATCH"):
                events.append({"REQ": rd.get("method"), "url": u[:200]})
        elif m.startswith("Network.responseReceived"):
            resp = msg["params"]["response"]
            u = resp.get("url","")
            if "api.hub.crunchdao.com" in u and resp.get("status") in (200, 201):
                events.append({"RESP": u[:200], "status": resp.get("status")})
    except websocket.WebSocketTimeoutException: pass
    except: pass
ws.settimeout(None)

print(f"\nNetwork events ({len(events)}):")
for e in events: print(" ", e)

# Final state
state2 = ev("({url: location.href, hint: document.body.innerText.slice(0,500)})", 99)
print(f"\nFinal URL: {state2.get('url')}")
ws.close()
