"""Trigger cloud run for a submission, arg = submission number."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

SUB = sys.argv[1] if len(sys.argv) > 1 else "5"
url = f"https://hub.crunchdao.com/competitions/datacrunch-2/models/12867/curly-crayfishwetminh-v2/runs/create?submissionNumber={SUB}"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req)
time.sleep(8)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if "runs/create" in t.get("url","") and f"submissionNumber={SUB}" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, method, params, mid):
    ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def evaluate(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True}, mid)
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
    const fakeEvt = new MouseEvent('click', {bubbles: true, cancelable: true});
    btn.dispatchEvent(fakeEvt);
    if (props && props.onClick) await props.onClick(fakeEvt);
    return {clicked: true};
  })()
""", 2)
print("CLICK:", r)

run_id = None
ws.settimeout(0.5)
end_time = time.time() + 15
while time.time() < end_time:
    try:
        msg = json.loads(ws.recv())
        if msg.get("method","").startswith("Network.responseReceived"):
            resp = msg["params"]["response"]
            u = resp.get("url","")
            if "/runs/" in u and "?_rsc" in u and resp.get("status") == 200:
                # capture run id
                seg = u.split("/runs/")[-1].split("?")[0]
                if seg.isdigit():
                    run_id = seg
    except websocket.WebSocketTimeoutException: pass
    except: pass
ws.settimeout(None)

state = evaluate(ws, "({url: location.href})", 100)
final = state.get('url') if state else None
print(f"Final URL: {final}")
if final and "/runs/" in final:
    run_id = final.split("/runs/")[-1].split("?")[0]
print(f"RUN_ID={run_id}")
ws.close()
