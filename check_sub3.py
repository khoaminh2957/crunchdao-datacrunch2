"""Check submission #3 status + trigger run if needed."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

def cdp(ws, method, params, mid):
    ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def evaluate(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True}, mid)
    return r.get("result", {}).get("result", {}).get("value")

# Open sub #3 page
url = "https://hub.crunchdao.com/competitions/datacrunch-2/projects/12867/curly-crayfishwetminh-v2/submissions/3"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req)
time.sleep(8)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
sub = [t for t in tabs if "submissions/3" in t.get("url","") and t.get("type") == "page"]
if not sub:
    print("NO_SUB3_TAB")
    sys.exit(1)

ws = websocket.create_connection(sub[0]["webSocketDebuggerUrl"], **WS_KW)
for i in range(15):
    time.sleep(1)
    n = evaluate(ws, "document.body.innerText.length", 1+i)
    if isinstance(n, int) and n > 1000: break

state = evaluate(ws, "({url: location.href, text: document.body.innerText.slice(0,4000)})", 100)
print(f"URL: {state.get('url')}")
print("=== TEXT ===")
print(state.get('text',''))
ws.close()
