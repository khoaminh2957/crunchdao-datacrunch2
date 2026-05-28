"""Create a NEW project — force fresh page each time."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

# Close ALL models/create tabs first
tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
for t in tabs:
    if "models/create" in t.get("url",""):
        try: urllib.request.urlopen(f"http://localhost:9222/json/close/{t['id']}")
        except: pass
time.sleep(2)

url = "https://hub.crunchdao.com/competitions/datacrunch-2/models/create"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(8)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
new_tabs = [t for t in tabs if "models/create" in t.get("url","") and t.get("type") == "page"]
if not new_tabs:
    print("NO CREATE TAB"); sys.exit(1)
tab = new_tabs[0]
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

# Read pre-filled name + click Create
r = ev(ws, r"""
  (() => {
    const nameField = document.querySelector('input[name="name"]');
    const btns = Array.from(document.querySelectorAll('button')).filter(b => b.offsetParent !== null && b.innerText.trim() === 'Create');
    if (!btns[0]) return {error: 'no Create btn'};
    const name = nameField?.value;
    btns[0].click();
    return {clicked: true, name};
  })()
""", 10)
print(f"CREATED NAME: {r.get('name')}")
time.sleep(4)

state = ev(ws, "({url: location.href})", 50)
print(f"After URL: {state.get('url')}")
ws.close()
