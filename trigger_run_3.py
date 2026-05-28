"""Trigger cloud run for submission #3."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

url = "https://hub.crunchdao.com/competitions/datacrunch-2/models/12867/curly-crayfishwetminh-v2/runs/create?submissionNumber=3"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req)
time.sleep(8)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if "runs/create" in t.get("url","") and "submissionNumber=3" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, method, params, mid):
    ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def evaluate(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True}, mid)
    if "exceptionDetails" in r.get("result", {}):
        print("JS_ERR:", json.dumps(r["result"]["exceptionDetails"])[:200])
    return r.get("result", {}).get("result", {}).get("value")

cdp(ws, "Network.enable", {}, 1)

r = evaluate(ws, r"""
  (async () => {
    const yes = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Yes I did');
    if (yes) yes.click();
    await new Promise(r => setTimeout(r, 500));
    const pw = document.querySelector('button[role="radio"][value="aws-cpu"]');
    if (pw) pw.click();
    await new Promise(r => setTimeout(r, 500));
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
    if (!btn) return {error: 'no btn'};
    const propsKey = Object.keys(btn).find(k => k.startsWith('__reactProps$'));
    const props = btn[propsKey];
    if (props) props.disabled = false;
    btn.disabled = false;
    btn.click();
    const fakeEvt = new MouseEvent('click', {bubbles: true, cancelable: true});
    btn.dispatchEvent(fakeEvt);
    if (props && props.onClick) await props.onClick(fakeEvt);
    return {clicked: true};
  })()
""", 2)
print("CLICK:", r)

events = []
ws.settimeout(0.5)
end_time = time.time() + 15
while time.time() < end_time:
    try:
        msg = json.loads(ws.recv())
        if msg.get("method","").startswith("Network.requestWillBeSent"):
            req_d = msg["params"]["request"]
            if "/runs" in req_d.get("url","") and req_d.get("method") == "POST":
                events.append({"url": req_d["url"], "postData": req_d.get("postData","")[:300]})
        elif msg.get("method","").startswith("Network.responseReceived"):
            resp = msg["params"]["response"]
            if "/runs" in resp.get("url","") and "/runs/" in resp.get("url","").rstrip("/"):
                events.append({"resp_url": resp["url"], "status": resp.get("status")})
    except websocket.WebSocketTimeoutException: pass
    except: pass
ws.settimeout(None)

print(f"\nNetwork events ({len(events)}):")
for e in events:
    print(f"  {e}")

state = evaluate(ws, "({url: location.href})", 100)
print(f"\nFinal URL: {state.get('url') if state else None}")
ws.close()
