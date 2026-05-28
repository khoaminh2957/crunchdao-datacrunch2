"""Trigger run for a submission on a non-default project."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

PROJECT = sys.argv[1]  # e.g. bewildered-chickadee
SUB = sys.argv[2]      # e.g. 1
url = f"https://hub.crunchdao.com/competitions/datacrunch-2/models/12867/{PROJECT}/runs/create?submissionNumber={SUB}"

# Close any existing
tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
for t in tabs:
    if f"runs/create" in t.get("url","") and PROJECT in t.get("url",""):
        try: urllib.request.urlopen(f"http://localhost:9222/json/close/{t['id']}")
        except: pass
time.sleep(2)

req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(10)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if "runs/create" in t.get("url","") and PROJECT in t.get("url","") and f"submissionNumber={SUB}" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, m, p, mid):
    ws.send(json.dumps({"id":mid,"method":m,"params":p or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def ev(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression":expr,"returnByValue":True,"awaitPromise":True}, mid)
    if "exceptionDetails" in r.get("result",{}):
        print("JS_ERR:", json.dumps(r["result"]["exceptionDetails"])[:200])
    return r.get("result",{}).get("result",{}).get("value")

cdp(ws, "Network.enable", {}, 1)

# Sequential clicks with verification
ev(ws, r"""(() => {const yes = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Yes I did'); if (yes) yes.click();})()""", 10)
time.sleep(2)
ev(ws, r"""(() => {const pw = document.querySelector('button[role="radio"][value="aws-cpu"]'); if (pw) pw.click();})()""", 11)
time.sleep(2)
r = ev(ws, r"""
  (async () => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
    if (!btn) return {error: 'no btn'};
    const propsKey = Object.keys(btn).find(k => k.startsWith('__reactProps$'));
    const props = btn[propsKey];
    btn.disabled = false;
    if (props) props.disabled = false;
    btn.click();
    await new Promise(r => setTimeout(r, 500));
    btn.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true, view: window}));
    await new Promise(r => setTimeout(r, 500));
    if (props && props.onClick) {
      try { await props.onClick({preventDefault: ()=>{}, stopPropagation: ()=>{}, type: 'click', target: btn}); } catch(e) {}
    }
    return {clicked: true};
  })()
""", 12)
print("CLICK:", r)

# Capture run id
run_id = None
ws.settimeout(0.5)
end_time = time.time() + 12
while time.time() < end_time:
    try:
        msg = json.loads(ws.recv())
        if msg.get("method","").startswith("Network.responseReceived"):
            u = msg["params"]["response"].get("url","")
            if "/runs/" in u and "?_rsc" in u and msg["params"]["response"].get("status") == 200:
                seg = u.split("/runs/")[-1].split("?")[0]
                if seg.isdigit():
                    run_id = seg
    except websocket.WebSocketTimeoutException: pass
    except: pass
ws.settimeout(None)

state = ev(ws, "({url: location.href, text: document.body.innerText.slice(0, 500)})", 99)
print(f"Final URL: {state.get('url')}")
if state and "/runs/" in state.get('url',''):
    run_id = state.get('url').split("/runs/")[-1].split("?")[0]
print(f"RUN_ID={run_id}")
ws.close()
