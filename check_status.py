"""Check CrunchDAO submission status + leaderboard via CDP."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

def cdp(ws, method, params, mid):
    ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid:
            return r

def evaluate(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate",
            {"expression": expr, "returnByValue": True, "awaitPromise": True}, mid)
    return r.get("result", {}).get("result", {}).get("value")

# Open submission page
sub_url = "https://hub.crunchdao.com/competitions/datacrunch-2/projects/12867/curly-crayfishwetminh-v2/submissions/1"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(sub_url)}", method="PUT")
urllib.request.urlopen(req)
time.sleep(8)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
sub_tab = [t for t in tabs if "submissions/1" in t.get("url","") and "crunchdao" in t.get("url","") and t.get("type") == "page"]
if sub_tab:
    ws = websocket.create_connection(sub_tab[0]["webSocketDebuggerUrl"], **WS_KW)
    for i in range(20):
        time.sleep(1)
        n = evaluate(ws, "document.body.innerText.length", 1+i)
        if isinstance(n, int) and n > 1000: break
    state = evaluate(ws, "({url: location.href, text: document.body.innerText.slice(0,4000)})", 100)
    print("=== Submission #1 page ===")
    print(state.get("text",""))
    ws.close()

# Open leaderboard
print("\n\n=== LEADERBOARD ===")
lb_url = "https://hub.crunchdao.com/competitions/datacrunch-2/leaderboard"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(lb_url)}", method="PUT")
urllib.request.urlopen(req)
time.sleep(8)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
lb_tab = [t for t in tabs if "leaderboard" in t.get("url","") and "crunchdao" in t.get("url","") and t.get("type") == "page"]
if lb_tab:
    ws = websocket.create_connection(lb_tab[0]["webSocketDebuggerUrl"], **WS_KW)
    for i in range(20):
        time.sleep(1)
        n = evaluate(ws, "document.body.innerText.length", 1+i)
        if isinstance(n, int) and n > 1000: break
    state = evaluate(ws, r"""
      (() => {
        const txt = document.body.innerText;
        // Look for our project name
        const ourName = 'curly-crayfishwetminh-v2';
        const idx = txt.indexOf(ourName);
        const context = idx > 0 ? txt.slice(Math.max(0,idx-200), idx+200) : null;
        return {url: location.href, has_our_entry: idx > 0, our_context: context, head: txt.slice(0,3000)};
      })()
    """, 100)
    print(json.dumps(state, indent=2, ensure_ascii=False)[:5000])
    ws.close()
