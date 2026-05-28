"""Dump the full leaderboard (no project filter)."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

url = "https://hub.crunchdao.com/competitions/datacrunch-2/leaderboard"

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
for t in tabs:
    if "leaderboard" in t.get("url","") and "datacrunch-2" in t.get("url",""):
        try: urllib.request.urlopen(f"http://localhost:9222/json/close/{t['id']}")
        except: pass
time.sleep(1)
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(10)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if "leaderboard" in t.get("url","") and "datacrunch-2" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, m, p, mid):
    ws.send(json.dumps({"id":mid,"method":m,"params":p or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def ev(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression":expr,"returnByValue":True,"awaitPromise":True}, mid)
    return r.get("result",{}).get("result",{}).get("value")

# Wait for table render
for i in range(10):
    time.sleep(1)
    n = ev(ws, "document.body.innerText.length", 1+i)
    if isinstance(n, int) and n > 5000: break

txt = ev(ws, "document.body.innerText", 99)
# Find leaderboard table content
idx = txt.find("Rank")
if idx > 0:
    print(txt[idx:idx+4000])
ws.close()
