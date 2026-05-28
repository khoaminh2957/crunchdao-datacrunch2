"""More robust trigger: opens NEW tab, longer waits between steps, verifies each."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

SUB = sys.argv[1]
url = f"https://hub.crunchdao.com/competitions/datacrunch-2/models/12867/curly-crayfishwetminh-v2/runs/create?submissionNumber={SUB}"

# Close existing create tabs for this sub
tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
for t in tabs:
    if f"runs/create" in t.get("url","") and f"submissionNumber={SUB}" in t.get("url",""):
        try:
            urllib.request.urlopen(f"http://localhost:9222/json/close/{t['id']}")
        except: pass

time.sleep(2)
# Open fresh
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req)
time.sleep(10)  # longer for SPA hydration

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if "runs/create" in t.get("url","") and f"submissionNumber={SUB}" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, m, p, mid):
    ws.send(json.dumps({"id":mid,"method":m,"params":p or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def ev(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression":expr,"returnByValue":True,"awaitPromise":True}, mid)
    if "exceptionDetails" in r.get("result",{}):
        print("JS_ERR:", r["result"]["exceptionDetails"])
    return r.get("result",{}).get("result",{}).get("value")

cdp(ws, "Network.enable", {}, 1)

# Step 1: Click Yes I did
r = ev(ws, r"""
  (() => {
    const yes = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Yes I did');
    if (!yes) return {error: 'no yes btn'};
    yes.click();
    return {clicked_yes: true};
  })()
""", 10)
print("STEP1:", r)
time.sleep(2)

# Step 2: Click Powerful radio
r = ev(ws, r"""
  (() => {
    const pw = document.querySelector('button[role="radio"][value="aws-cpu"]');
    if (!pw) return {error: 'no radio'};
    pw.click();
    // Verify
    return {clicked_radio: true, aria: pw.getAttribute('aria-checked')};
  })()
""", 11)
print("STEP2:", r)
time.sleep(2)

# Step 3: Click Create run via React props (force)
r = ev(ws, r"""
  (async () => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
    if (!btn) return {error: 'no create btn'};
    const propsKey = Object.keys(btn).find(k => k.startsWith('__reactProps$'));
    const props = btn[propsKey];
    btn.disabled = false;
    if (props) props.disabled = false;
    // Native click
    btn.click();
    await new Promise(r => setTimeout(r, 500));
    // Synthetic click event
    btn.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true, view: window}));
    await new Promise(r => setTimeout(r, 500));
    // Direct call onClick handler
    if (props && props.onClick) {
      try { await props.onClick({preventDefault: ()=>{}, stopPropagation: ()=>{}, type: 'click', target: btn}); } catch(e) { return {error: 'onClick threw: ' + e.message}; }
    }
    return {clicked_create: true, disabled: btn.disabled, has_onClick: !!(props?.onClick)};
  })()
""", 12)
print("STEP3:", r)

# Wait + capture network events
events = []
ws.settimeout(0.5)
end_time = time.time() + 15
while time.time() < end_time:
    try:
        msg = json.loads(ws.recv())
        if msg.get("method","").startswith("Network.requestWillBeSent"):
            req_d = msg["params"]["request"]
            if "/runs" in req_d.get("url","") and req_d.get("method") == "POST":
                events.append({"POST": req_d["url"], "payload": req_d.get("postData","")[:200]})
        elif msg.get("method","").startswith("Network.responseReceived"):
            resp = msg["params"]["response"]
            u = resp.get("url","")
            if "/runs/" in u and "?_rsc" in u:
                seg = u.split("/runs/")[-1].split("?")[0]
                if seg.isdigit():
                    events.append({"NEW_RUN_ID": seg, "status": resp.get("status")})
    except websocket.WebSocketTimeoutException: pass
    except: pass
ws.settimeout(None)

print(f"\nNetwork ({len(events)}):")
for e in events: print(" ", e)

state = ev(ws, "({url: location.href})", 99)
print(f"\nFinal URL: {state.get('url') if state else None}")
ws.close()
