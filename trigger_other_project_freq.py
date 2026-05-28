"""Trigger run on a non-default project with custom Train Frequency."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

PROJECT = sys.argv[1]
SUB = sys.argv[2]
FREQ = sys.argv[3] if len(sys.argv) > 3 else "1"

url = f"https://hub.crunchdao.com/competitions/datacrunch-2/models/12867/{PROJECT}/runs/create?submissionNumber={SUB}"

# Close existing tabs for sub
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
        print("JS_ERR:", json.dumps(r["result"]["exceptionDetails"])[:300])
    return r.get("result",{}).get("result",{}).get("value")

cdp(ws, "Network.enable", {}, 1)

# Click Yes I did
ev(ws, """(() => {const yes = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Yes I did'); if (yes) yes.click();})()""", 10)
time.sleep(2)
# Click Powerful radio (aws-cpu)
ev(ws, """(() => {const pw = document.querySelector('button[role="radio"][value="aws-cpu"]'); if (pw) pw.click();})()""", 11)
time.sleep(2)

# Set Train Frequency to FREQ (find numeric input)
r = ev(ws, f"""
  (() => {{
    const inputs = Array.from(document.querySelectorAll('input[type="number"], input[type="text"]'));
    // Find one with value '0' (current freq) or label nearby
    const freqInput = inputs.find(i => i.value === '0' && i.offsetParent !== null);
    if (!freqInput) return {{error: 'no freq input', inputs: inputs.map(i => ({{type: i.type, value: i.value, name: i.name}}))}};
    // Native setter so React detects
    const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
    nativeSetter.call(freqInput, '{FREQ}');
    freqInput.dispatchEvent(new Event('input', {{bubbles: true}}));
    freqInput.dispatchEvent(new Event('change', {{bubbles: true}}));
    return {{set: true, current: freqInput.value}};
  }})()
""", 12)
print("SET FREQ:", r)
time.sleep(2)

# Click Create run
r = ev(ws, r"""
  (async () => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run');
    if (!btn) return {error: 'no Create run btn'};
    const propsKey = Object.keys(btn).find(k => k.startsWith('__reactProps$'));
    const props = btn[propsKey];
    btn.disabled = false;
    if (props) props.disabled = false;
    btn.click();
    await new Promise(r => setTimeout(r, 500));
    btn.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true, view: window}));
    await new Promise(r => setTimeout(r, 500));
    if (props && props.onClick) { try { await props.onClick({preventDefault: ()=>{}, stopPropagation: ()=>{}, type: 'click', target: btn}); } catch(e) {} }
    return {clicked: true};
  })()
""", 13)
print("CLICK:", r)

# Capture payload to confirm freq value sent
events = []
ws.settimeout(0.5)
end_time = time.time() + 12
while time.time() < end_time:
    try:
        msg = json.loads(ws.recv())
        if msg.get("method","").startswith("Network.requestWillBeSent"):
            rd = msg["params"]["request"]
            u = rd.get("url","")
            if "/runs" in u and rd.get("method") == "POST" and "api.hub.crunchdao" in u:
                events.append({"url": u, "payload": rd.get("postData","")[:300]})
    except websocket.WebSocketTimeoutException: pass
    except: pass
ws.settimeout(None)
for e in events: print(" PAYLOAD:", e.get("payload"))

state = ev(ws, "({url: location.href})", 99)
print(f"Final URL: {state.get('url')}")
ws.close()
