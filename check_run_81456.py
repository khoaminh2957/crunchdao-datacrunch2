"""Verify run 81456 status."""
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

run_url = "https://hub.crunchdao.com/competitions/datacrunch-2/models/wet-minh/curly-crayfishwetminh-v2/runs/81456"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(run_url)}", method="PUT")
urllib.request.urlopen(req)
time.sleep(8)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if "runs/81456" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

for i in range(15):
    time.sleep(1)
    n = evaluate(ws, "document.body.innerText.length", 1+i)
    if isinstance(n, int) and n > 1500: break

# Call API direct to get run details
r = evaluate(ws, r"""
  (async () => {
    const res = await fetch('https://api.hub.crunchdao.com/v3/competitions/datacrunch-2/projects/12867/curly-crayfishwetminh-v2/runs/81456', {credentials: 'include'});
    if (res.ok) {
      return await res.json();
    }
    return {status: res.status, body: (await res.text()).slice(0,500)};
  })()
""", 100)
print("RUN 81456 API:")
print(json.dumps(r, indent=2, ensure_ascii=False)[:3000])

print("\n=== Visible page ===")
state = evaluate(ws, "({url: location.href, text: document.body.innerText.slice(0,2500)})", 101)
print(state.get('text', '')[:2500])
ws.close()
