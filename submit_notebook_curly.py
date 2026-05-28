"""Submit v60.ipynb via Notebook tab. Model = curly. Use exact flow that worked before."""
import json, urllib.request, urllib.parse, websocket, time, sys, os
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

NOTEBOOK = r"C:\Users\Admin\earn5usd\crunchdao\v60_submission_files\v60.ipynb"
PROJECT = "curly-crayfishwetminh-v2"
MESSAGE = "v60 via UI: XGB 500/d6/lr=0.03 RAW seed=42 (cloud prev 0.0806)"

# Close all relevant tabs
tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
for t in tabs:
    u = t.get("url","")
    if "submit/notebook" in u or "submit/files" in u or "Processing Notebook" in t.get("title",""):
        try: urllib.request.urlopen(f"http://localhost:9222/json/close/{t['id']}")
        except: pass
time.sleep(3)

# Open fresh notebook submit for curly
url = f"https://hub.crunchdao.com/competitions/datacrunch-2/submit/notebook?projectName={PROJECT}"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(10)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if "submit/notebook" in t.get("url","") and "curly" in t.get("url","") and t.get("type") == "page"][-1]
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

# Verify model = curly
st = ev("""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b =>
      /curly|bewildered|historic|efficient/.test(b.innerText) && b.innerText.length < 50
    );
    return {model: btn?.innerText.trim()};
  })()
""", 5)
print(f"Initial model: {st}")

# If not curly, click dropdown + select curly
if st and "curly" not in (st.get("model") or ""):
    print("Switching dropdown to curly...")
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

st2 = ev("""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b =>
      /curly|bewildered|historic|efficient/.test(b.innerText) && b.innerText.length < 50
    );
    const subBtn = Array.from(document.querySelectorAll('button')).find(b => /Submit to/.test(b.innerText));
    return {model: btn?.innerText.trim(), submit_text: subBtn?.innerText.trim()};
  })()
""", 12)
print(f"After: {st2}")

# Find file input + upload
doc = cdp("DOM.getDocument", {}, 20)
root_id = doc["result"]["root"]["nodeId"]
file_inputs = cdp("DOM.querySelectorAll", {"nodeId": root_id, "selector": 'input[type="file"]'}, 21)
node_ids = file_inputs["result"]["nodeIds"]
print(f"File inputs: {node_ids}")

cdp("DOM.setFileInputFiles", {"files": [NOTEBOOK], "nodeId": node_ids[0]}, 50)
time.sleep(2)

# Dispatch change
ev("""
  (() => {
    const inp = document.querySelector('input[type="file"]');
    inp.dispatchEvent(new Event('change', {bubbles: true}));
  })()
""", 60)
time.sleep(2)

# Fill message
ev(f"""
  (() => {{
    const txt = document.querySelector('textarea');
    if (txt) {{
      const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
      setter.call(txt, {json.dumps(MESSAGE)});
      txt.dispatchEvent(new Event('input', {{bubbles: true}}));
    }}
  }})()
""", 70)
time.sleep(2)

# Click submit
r = ev("""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => /Submit to/.test(b.innerText) && b.offsetParent !== null);
    if (!btn) return {error: 'no submit'};
    btn.click();
    return {clicked: true, text: btn.innerText.trim()};
  })()
""", 80)
print(f"Submit click: {r}")

# Wait for processing/result
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

state = ev("({url: location.href, hint: document.body.innerText.slice(0,1500)})", 99)
print(f"\nFinal URL: {state.get('url')}")
# Look for "Processing Notebook" indicator
print(f"Page has 'Processing'? {'Processing' in (state.get('hint') or '')}")
ws.close()
