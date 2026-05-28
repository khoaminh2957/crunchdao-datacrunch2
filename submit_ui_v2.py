"""Submit v60.ipynb via UI — verify each step carefully + capture network."""
import json, urllib.request, urllib.parse, websocket, time, sys, os
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

PROJECT = "bewildered-chickadee"
NOTEBOOK = r"C:\Users\Admin\earn5usd\crunchdao\v60_submission_files\v60.ipynb"

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
for t in tabs:
    if "submit/notebook" in t.get("url","") and PROJECT in t.get("url",""):
        try: urllib.request.urlopen(f"http://localhost:9222/json/close/{t['id']}")
        except: pass
time.sleep(2)

url = f"https://hub.crunchdao.com/competitions/datacrunch-2/submit/notebook?projectName={PROJECT}"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(10)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if "submit/notebook" in t.get("url","") and PROJECT in t.get("url","") and t.get("type") == "page"][0]
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
cdp("Page.enable", {}, 2)
cdp("DOM.enable", {}, 3)

# Find file input via DOM
doc = cdp("DOM.getDocument", {}, 20)
root_id = doc["result"]["root"]["nodeId"]
file_inputs = cdp("DOM.querySelectorAll", {"nodeId": root_id, "selector": 'input[type="file"]'}, 21)
node_ids = file_inputs["result"]["nodeIds"]
print(f"File input nodeIds: {node_ids}")

# Set file
cdp("DOM.setFileInputFiles", {"files": [NOTEBOOK], "nodeId": node_ids[0]}, 50)
time.sleep(2)

# Verify
fcheck = ev("""
  (() => {
    const inp = document.querySelector('input[type="file"]');
    return {n_files: inp.files.length, name: inp.files[0]?.name, size: inp.files[0]?.size};
  })()
""", 60)
print("File set:", fcheck)

# Wait for React to detect file (may need to dispatch change event)
ev("""
  (() => {
    const inp = document.querySelector('input[type="file"]');
    inp.dispatchEvent(new Event('change', {bubbles: true}));
  })()
""", 65)
time.sleep(2)

# Check submit button state
btn_state = ev("""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => /Submit to/.test(b.innerText));
    if (!btn) return {error: 'no btn'};
    const propsKey = Object.keys(btn).find(k => k.startsWith('__reactProps$'));
    return {
      text: btn.innerText,
      disabled: btn.disabled,
      react_disabled: btn[propsKey]?.disabled,
      type: btn.type,
      form: btn.form?.id || btn.closest('form')?.id || 'no form',
    };
  })()
""", 70)
print("Submit btn state:", btn_state)

# Wait a few seconds for any validation, then check again
time.sleep(3)
btn_state2 = ev("""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => /Submit to/.test(b.innerText));
    return {disabled: btn.disabled, react_disabled: Object.keys(btn).find(k => k.startsWith('__reactProps$')) ? btn[Object.keys(btn).find(k => k.startsWith('__reactProps$'))]?.disabled : 'na'};
  })()
""", 71)
print("Btn after wait:", btn_state2)

# Click + capture network
events = []
ws.settimeout(0.1)
end_check = time.time() + 1
while time.time() < end_check:
    try: ws.recv()
    except: pass
ws.settimeout(None)

r = ev("""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => /Submit to/.test(b.innerText));
    if (!btn) return {error: 'no btn'};
    btn.click();
    return {clicked: true, disabled: btn.disabled};
  })()
""", 80)
print("Click:", r)

ws.settimeout(0.5)
end_time = time.time() + 20
while time.time() < end_time:
    try:
        msg = json.loads(ws.recv())
        m = msg.get("method","")
        if m.startswith("Network.requestWillBeSent"):
            rd = msg["params"]["request"]
            u = rd.get("url","")
            if "api.hub.crunchdao.com" in u and rd.get("method") in ("POST","PUT"):
                events.append({"REQ": rd.get("method"), "url": u[:200], "payload_len": len(rd.get("postData","")), "payload_head": rd.get("postData","")[:200]})
        elif m.startswith("Network.responseReceived"):
            resp = msg["params"]["response"]
            u = resp.get("url","")
            if "api.hub.crunchdao.com" in u:
                events.append({"RESP": u[:200], "status": resp.get("status")})
    except websocket.WebSocketTimeoutException: pass
    except: pass
ws.settimeout(None)
print(f"\nNetwork events ({len(events)}):")
for e in events:
    print(" ", e)

# Final page state
state = ev("({url: location.href, text: document.body.innerText.slice(-2000)})", 99)
print(f"\nFinal URL: {state.get('url')}")
print("Page tail:", state.get('text','')[-500:])
ws.close()
