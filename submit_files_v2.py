"""Submit files: wait for DOM refresh after model change before uploading."""
import json, urllib.request, urllib.parse, websocket, time, sys, os
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

MAIN_PY = r"C:\Users\Admin\earn5usd\crunchdao\v60_submission_files\main.py"
REQ_TXT = r"C:\Users\Admin\earn5usd\crunchdao\v60_submission_files\requirements.txt"
TARGET_PROJECT = "curly-crayfishwetminh-v2"

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
# Close existing files submission tabs
for t in tabs:
    if "submit/files" in t.get("url",""):
        try: urllib.request.urlopen(f"http://localhost:9222/json/close/{t['id']}")
        except: pass
time.sleep(2)

# Open fresh directly with curly project
url = f"https://hub.crunchdao.com/competitions/datacrunch-2/submit/files?projectName={TARGET_PROJECT}"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(10)

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

# Verify model is already curly (URL param should set it)
state = ev("""
  (() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const modelBtn = buttons.find(b => /curly|bewildered|historic|efficient/.test(b.innerText) && b.innerText.length < 50);
    const fileInputs = document.querySelectorAll('input[type="file"]').length;
    return {model: modelBtn?.innerText.trim(), file_inputs: fileInputs};
  })()
""", 5)
print("Initial state:", state)

# If model not curly, change it
if state and "curly" not in (state.get("model") or ""):
    print("Need to change model to curly...")
    ev("""
      (() => {
        const btns = Array.from(document.querySelectorAll('button'));
        const sel = btns.find(b => /bewildered|historic|efficient/.test(b.innerText) && b.innerText.length < 50);
        if (sel) sel.click();
      })()
    """, 10)
    time.sleep(2)
    ev("""
      (() => {
        const items = Array.from(document.querySelectorAll('[role="option"], button, div, a')).filter(e => e.offsetParent !== null);
        const curly = items.find(e => e.innerText.trim() === 'curly-crayfishwetminh-v2');
        if (curly) curly.click();
      })()
    """, 11)
    time.sleep(3)

# Re-query DOM for file inputs
doc = cdp("DOM.getDocument", {}, 20)
root_id = doc["result"]["root"]["nodeId"]
file_inputs = cdp("DOM.querySelectorAll", {"nodeId": root_id, "selector": 'input[type="file"]'}, 21)
node_ids = file_inputs["result"]["nodeIds"]
print(f"File input nodeIds: {node_ids}")

if not node_ids:
    print("ERROR: no file inputs. Page state?")
    state2 = ev("({url: location.href, sample: document.body.innerText.slice(0,1000)})", 30)
    print(state2)
    ws.close()
    sys.exit(1)

# Set files
cdp("DOM.setFileInputFiles", {"files": [MAIN_PY, REQ_TXT], "nodeId": node_ids[0]}, 50)
time.sleep(2)

fcheck = ev("""
  (() => {
    const inp = document.querySelector('input[type="file"]');
    return {n_files: inp.files.length, names: Array.from(inp.files).map(f => f.name)};
  })()
""", 60)
print("Files set:", fcheck)
ev("""
  (() => {
    const inp = document.querySelector('input[type="file"]');
    inp.dispatchEvent(new Event('change', {bubbles: true}));
  })()
""", 61)
time.sleep(2)

# Check submit button + Project name
btn_state = ev("""
  (() => {
    const btns = Array.from(document.querySelectorAll('button')).filter(b => b.offsetParent !== null);
    const submit = btns.find(b => /Submit to/.test(b.innerText));
    return {
      submit_text: submit?.innerText.trim(),
      submit_disabled: submit?.disabled,
    };
  })()
""", 70)
print("Submit btn:", btn_state)

# Click submit
ev("""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => /Submit to/.test(b.innerText) && b.offsetParent !== null);
    if (btn) btn.click();
  })()
""", 80)

events = []
ws.settimeout(0.5)
end_time = time.time() + 25
while time.time() < end_time:
    try:
        msg = json.loads(ws.recv())
        m = msg.get("method","")
        if m.startswith("Network.requestWillBeSent"):
            rd = msg["params"]["request"]
            u = rd.get("url","")
            if "api.hub.crunchdao.com" in u and rd.get("method") in ("POST","PUT"):
                events.append({"REQ": rd.get("method"), "url": u[:200]})
        elif m.startswith("Network.responseReceived"):
            resp = msg["params"]["response"]
            u = resp.get("url","")
            if "api.hub.crunchdao.com" in u:
                events.append({"RESP": u[:200], "status": resp.get("status")})
    except websocket.WebSocketTimeoutException: pass
    except: pass
ws.settimeout(None)
print(f"\nNetwork ({len(events)}):")
for e in events: print(" ", e)

state2 = ev("({url: location.href, hint: document.body.innerText.slice(0,500)})", 99)
print(f"Final URL: {state2.get('url')}")
print(f"Page hint: {state2.get('hint','')[:300]}")
ws.close()
