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

# Open run page + reload
run_url = "https://hub.crunchdao.com/competitions/datacrunch-2/models/wet-minh/curly-crayfishwetminh-v2/runs/81456"
tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
existing = [t for t in tabs if "runs/81456" in t.get("url","") and t.get("type") == "page"]
if not existing:
    req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(run_url)}", method="PUT")
    urllib.request.urlopen(req)
    time.sleep(8)
    tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
    existing = [t for t in tabs if "runs/81456" in t.get("url","") and t.get("type") == "page"]

if existing:
    ws = websocket.create_connection(existing[0]["webSocketDebuggerUrl"], **WS_KW)
    cdp(ws, "Page.enable", {}, 0)
    cdp(ws, "Page.reload", {}, 1)
    time.sleep(7)
    for i in range(10):
        time.sleep(1)
        n = evaluate(ws, "document.body.innerText.length", 2+i)
        if isinstance(n, int) and n > 1500: break
    txt = evaluate(ws, "document.body.innerText", 100) or ""
    # Extract key fields
    import re
    status = re.search(r"Status\s*\n\s*(\w+)", txt)
    duration = re.search(r"Duration\s*\n\s*([^\n]+)", txt)
    score = re.search(r"(Score|score)[\s\n:]+([0-9.]+)", txt)
    print(f"Run 81456:")
    print(f"  Status: {status.group(1) if status else '?'}")
    print(f"  Duration: {duration.group(1) if duration else '?'}")
    print(f"  Score: {score.group(2) if score else '(not yet)'}")
    print(f"  Full text excerpt:\n{txt[:2500]}")
    ws.close()

# Check leaderboard
print("\n\n=== LEADERBOARD MY RANK ===")
lb_url = "https://hub.crunchdao.com/competitions/datacrunch-2/leaderboard?projectName=curly-crayfishwetminh-v2"
tabs2 = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
lb = [t for t in tabs2 if "leaderboard" in t.get("url","") and t.get("type") == "page"]
if lb:
    ws2 = websocket.create_connection(lb[0]["webSocketDebuggerUrl"], **WS_KW)
    cdp(ws2, "Page.enable", {}, 0)
    cdp(ws2, "Page.navigate", {"url": lb_url}, 1)
    time.sleep(7)
    for i in range(10):
        time.sleep(1)
        n = evaluate(ws2, "document.body.innerText.length", 2+i)
        if isinstance(n, int) and n > 1500: break
    state = evaluate(ws2, r"""
      (() => {
        const txt = document.body.innerText;
        const rankMatch = txt.match(/MY RANK\s*\n+\s*(\S+)/);
        const scoreMatch = txt.match(/MY SCORE\s*\n+\s*(\S+)/);
        const ourIdx = txt.indexOf('curly-crayfishwetminh-v2');
        return {my_rank: rankMatch?.[1], my_score: scoreMatch?.[1], our_context: ourIdx > 0 ? txt.slice(Math.max(0,ourIdx-100), ourIdx+400) : null};
      })()
    """, 100)
    print(json.dumps(state, indent=2, ensure_ascii=False)[:2500])
    ws2.close()
