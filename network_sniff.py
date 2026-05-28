"""Use CDP Network domain to capture all requests during form interaction."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

# Navigate fresh to the runs/create page
url = "https://hub.crunchdao.com/competitions/datacrunch-2/models/12867/curly-crayfishwetminh-v2/runs/create?submissionNumber=1"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req)
time.sleep(8)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if "runs/create" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

def cdp_send(method, params, mid):
    ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))

def cdp_wait(mid):
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

# Enable Network + Runtime
cdp_send("Network.enable", {}, 1); cdp_wait(1)
cdp_send("Runtime.enable", {}, 2); cdp_wait(2)

# Collect network events
events = []
def collect_events(duration=15):
    end = time.time() + duration
    ws.settimeout(0.5)
    while time.time() < end:
        try:
            msg = json.loads(ws.recv())
            if msg.get("method", "").startswith("Network.requestWillBeSent"):
                req = msg["params"]["request"]
                url = req.get("url", "")
                if "crunchdao" in url and "_next" not in url and not url.endswith((".js",".css",".woff2",".png",".svg",".webp",".ico")):
                    events.append({"url": url, "method": req.get("method"), "postData": (req.get("postData") or "")[:300], "headers": {k:v for k,v in req.get("headers",{}).items() if k.lower() in ("content-type","next-action","next-router-state-tree","cookie","authorization") and len(v) < 200}})
        except websocket.WebSocketTimeoutException: pass
        except: pass
    ws.settimeout(None)

# Set up form fields via API + click Create run + sniff
cdp_send("Runtime.evaluate", {"expression": r"""
  (async () => {
    // 1. Click Yes I did
    const yes = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Yes I did');
    if (yes) yes.click();
    await new Promise(r => setTimeout(r, 500));
    // 2. Re-select Powerful
    const pw = document.querySelector('button[role="radio"][value="aws-cpu"]');
    if (pw) pw.click();
    await new Promise(r => setTimeout(r, 500));
    return {ok: true};
  })()
""", "returnByValue": True, "awaitPromise": True}, 10)
cdp_wait(10)
print("Form prepared")
time.sleep(2)

# Now click Create run via React props (force enable)
cdp_send("Runtime.evaluate", {"expression": r"""
  (async () => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
    if (!btn) return {error: 'no btn'};
    const propsKey = Object.keys(btn).find(k => k.startsWith('__reactProps$'));
    const props = btn[propsKey];
    if (props) props.disabled = false;
    btn.disabled = false;
    btn.removeAttribute('disabled');
    // Dispatch click via multiple methods
    btn.click();
    const fakeEvt = new MouseEvent('click', {bubbles: true, cancelable: true});
    btn.dispatchEvent(fakeEvt);
    if (props && props.onClick) await props.onClick(fakeEvt);
    return {clicked: true};
  })()
""", "returnByValue": True, "awaitPromise": True}, 20)
cdp_wait(20)
print("Create run clicked — collecting network events for 15s")
collect_events(15)

print(f"\nCaptured {len(events)} relevant requests:")
for e in events:
    print(json.dumps(e, ensure_ascii=False, indent=2))
ws.close()
