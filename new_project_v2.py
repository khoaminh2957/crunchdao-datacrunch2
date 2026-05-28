"""Create a fresh 3rd CrunchDAO project for parallel quota."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

# Navigate directly to models/create page
url = "https://hub.crunchdao.com/competitions/datacrunch-2/models/create"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(8)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if "models/create" in t.get("url","") and t.get("type") == "page"][-1]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, m, p, mid):
    ws.send(json.dumps({"id":mid,"method":m,"params":p or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def ev(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression":expr,"returnByValue":True,"awaitPromise":True}, mid)
    return r.get("result",{}).get("result",{}).get("value")

cdp(ws, "Network.enable", {}, 1)

# Read name + click Create
r = ev(ws, r"""
  (() => {
    const nameField = document.querySelector('input[name="name"]');
    const btn = Array.from(document.querySelectorAll('button')).filter(b => b.offsetParent !== null && b.innerText.trim() === 'Create');
    if (!btn[0]) return {error: 'no Create btn', text: document.body.innerText.slice(0,500)};
    const name = nameField?.value || 'auto-name';
    btn[0].click();
    return {clicked: true, name};
  })()
""", 10)
print("CLICK:", r)
time.sleep(3)

state = ev(ws, "({url: location.href, text: document.body.innerText.slice(0,800)})", 50)
print(f"\nAfter URL: {state.get('url')}")
print(f"Hint: {state.get('text','')[:500]}")
ws.close()
