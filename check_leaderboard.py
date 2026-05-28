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

lb_url = "https://hub.crunchdao.com/competitions/datacrunch-2/leaderboard?projectName=curly-crayfishwetminh-v2"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(lb_url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(8)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
lb = sorted([t for t in tabs if "leaderboard" in t.get("url","") and "curly-crayfishwetminh-v2" in t.get("url","") and t.get("type") == "page"], key=lambda t: 0)[0]
ws = websocket.create_connection(lb["webSocketDebuggerUrl"], **WS_KW)

for i in range(15):
    time.sleep(1)
    n = evaluate(ws, "document.body.innerText.length", 1+i)
    if isinstance(n, int) and n > 2000: break

state = evaluate(ws, r"""
  (() => {
    const txt = document.body.innerText;
    const rankMatch = txt.match(/MY RANK\s*\n+\s*([\S-]+)/);
    const scoreMatch = txt.match(/MY SCORE\s*\n+\s*([\S-]+)/);
    return {my_rank: rankMatch?.[1], my_score: scoreMatch?.[1], full: txt.slice(0,5000)};
  })()
""", 100)
print(f"MY_RANK: {state.get('my_rank')}")
print(f"MY_SCORE: {state.get('my_score')}")
print()
# Find our row in leaderboard
full = state.get('full','')
idx = full.find('curly-crayfishwetminh-v2')
if idx > 0:
    print("=== Context around our entry ===")
    print(full[max(0,idx-50):idx+800])
ws.close()
