"""Navigate to the CrunchDAO 'create new model' page and dump DOM + form structure."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

url = "https://hub.crunchdao.com/competitions/datacrunch-2/submit"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(8)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tabs = [t for t in tabs if "datacrunch-2/submit" in t.get("url","") and t.get("type") == "page"]
if not tabs:
    print("NO TAB"); sys.exit(1)
ws = websocket.create_connection(tabs[0]["webSocketDebuggerUrl"], **WS_KW)

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

# Capture network for the model-creation POST
cdp(ws, "Network.enable", {}, 1)

state = ev(ws, r"""
({
  url: location.href,
  text: document.body.innerText.slice(0, 4000),
  buttons: Array.from(document.querySelectorAll('button')).slice(0, 20).map(b => ({txt: b.innerText.trim(), disabled: b.disabled}))
})
""", 99)
print("=== SUBMIT PAGE ===")
print(json.dumps(state, indent=2, ensure_ascii=False)[:5000])
ws.close()
