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

# Open leaderboard with our model filter
url = "https://hub.crunchdao.com/competitions/datacrunch-2/leaderboard?projectName=curly-crayfishwetminh-v2"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req)
time.sleep(8)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
lb = [t for t in tabs if "leaderboard" in t.get("url","") and "curly-crayfishwetminh-v2" in t.get("url","") and t.get("type") == "page"]
if lb:
    ws = websocket.create_connection(lb[0]["webSocketDebuggerUrl"], **WS_KW)
    for i in range(15):
        time.sleep(1)
        n = evaluate(ws, "document.body.innerText.length", 1+i)
        if isinstance(n, int) and n > 1500: break
    state = evaluate(ws, r"""
      (() => {
        const txt = document.body.innerText;
        const ourIdx = txt.indexOf('curly-crayfishwetminh-v2');
        const rank = txt.match(/MY RANK[\s\n]+([0-9]+|-)/);
        const score = txt.match(/MY SCORE[\s\n]+([0-9.]+|-)/);
        return {url: location.href, my_rank: rank?.[1], my_score: score?.[1], our_context: ourIdx > 0 ? txt.slice(Math.max(0,ourIdx-100), ourIdx+500) : null};
      })()
    """, 100)
    print(json.dumps(state, indent=2, ensure_ascii=False)[:3000])
    ws.close()

# Also check submission #1 page
print("\n\n=== Submission #1 details ===")
url2 = "https://hub.crunchdao.com/competitions/datacrunch-2/projects/12867/curly-crayfishwetminh-v2/submissions/1"
tabs2 = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
sub = [t for t in tabs2 if "submissions/1" in t.get("url","") and t.get("type") == "page"]
if sub:
    ws2 = websocket.create_connection(sub[0]["webSocketDebuggerUrl"], **WS_KW)
    cdp(ws2, "Page.enable", {}, 0)
    cdp(ws2, "Page.reload", {}, 1)
    time.sleep(7)
    state2 = evaluate(ws2, "({url: location.href, text: document.body.innerText.slice(0,3000)})", 100)
    print(state2.get('text',''))
    ws2.close()
