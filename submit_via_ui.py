"""Submit v60.ipynb via Notebook Submission form using CDP file upload."""
import json, urllib.request, urllib.parse, websocket, time, sys, os
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

PROJECT = "curly-crayfishwetminh-v2"
NOTEBOOK = r"C:\Users\Admin\earn5usd\crunchdao\v60_submission_files\v60.ipynb"
MESSAGE = "v60 re-submit via UI: XGB 500/d6/lr=0.03 RAW seed=42 (prev cloud 0.0806)"

assert os.path.exists(NOTEBOOK), f"Missing: {NOTEBOOK}"
print(f"Notebook: {NOTEBOOK} ({os.path.getsize(NOTEBOOK)} bytes)")

# Close existing submit tabs
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

# Enable required domains
cdp("Page.enable", {}, 1)
cdp("DOM.enable", {}, 2)
cdp("Network.enable", {}, 3)

# Find notebook file input via JS, get its backendNodeId
node_info = ev("""
  (() => {
    const inputs = Array.from(document.querySelectorAll('input[type="file"]'));
    return inputs.map((el, i) => ({
      i, accept: el.accept, multiple: el.multiple, id: el.id, name: el.name,
    }));
  })()
""", 10)
print("File inputs:", node_info)

# Use DOM.querySelector to get backendNodeId
doc = cdp("DOM.getDocument", {}, 20)
root_id = doc["result"]["root"]["nodeId"]
file_inputs = cdp("DOM.querySelectorAll", {"nodeId": root_id, "selector": 'input[type="file"]'}, 21)
node_ids = file_inputs["result"]["nodeIds"]
print(f"File input nodeIds: {node_ids}")

# For each, get backendNodeId and check accept attr
for nid in node_ids:
    desc = cdp("DOM.describeNode", {"nodeId": nid}, 30 + nid)
    node = desc["result"]["node"]
    print(f"nodeId={nid} attrs: {dict(zip(node['attributes'][::2], node['attributes'][1::2]))}")

# The first input is the notebook (.ipynb), second is model files
# Set file path
res = cdp("DOM.setFileInputFiles", {
    "files": [NOTEBOOK],
    "nodeId": node_ids[0]
}, 50)
print("setFileInputFiles result:", res)
time.sleep(2)

# Check file was set
fcheck = ev("""
  (() => {
    const inp = document.querySelector('input[type="file"]');
    return {n_files: inp.files.length, name: inp.files[0]?.name, size: inp.files[0]?.size};
  })()
""", 60)
print("File check:", fcheck)

# Fill message
ev(f"""
  (() => {{
    const txt = document.querySelector('textarea');
    if (txt) {{
      const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
      setter.call(txt, {json.dumps(MESSAGE)});
      txt.dispatchEvent(new Event('input', {{bubbles: true}}));
      txt.dispatchEvent(new Event('change', {{bubbles: true}}));
    }}
  }})()
""", 70)
time.sleep(2)

# Click Submit button
r = ev(f"""
  (() => {{
    const btns = Array.from(document.querySelectorAll('button')).filter(b => /Submit to/.test(b.innerText) && b.offsetParent !== null);
    if (btns.length === 0) return {{error: 'no submit btn'}};
    const btn = btns[0];
    console.log('disabled:', btn.disabled);
    btn.disabled = false;
    const propsKey = Object.keys(btn).find(k => k.startsWith('__reactProps$'));
    if (propsKey && btn[propsKey]) btn[propsKey].disabled = false;
    btn.click();
    btn.dispatchEvent(new MouseEvent('click', {{bubbles: true, cancelable: true, view: window}}));
    return {{clicked: true, text: btn.innerText}};
  }})()
""", 80)
print("Submit click:", r)

# Wait for response
events = []
ws.settimeout(0.5)
end_time = time.time() + 15
while time.time() < end_time:
    try:
        msg = json.loads(ws.recv())
        if msg.get("method","").startswith("Network.responseReceived"):
            u = msg["params"]["response"].get("url","")
            if "submissions" in u and "api.hub.crunchdao.com" in u:
                events.append({"url": u, "status": msg["params"]["response"].get("status")})
    except websocket.WebSocketTimeoutException: pass
    except: pass
ws.settimeout(None)
for e in events: print("API:", e)

state = ev("({url: location.href, hint: document.body.innerText.slice(0, 800)})", 99)
print(f"\nFinal URL: {state.get('url')}")
ws.close()
