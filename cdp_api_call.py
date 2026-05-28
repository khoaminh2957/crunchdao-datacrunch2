"""Call API from inside CDP browser (has session cookies)."""
import json, urllib.request, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)
tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if "hub.crunchdao.com" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, method, params, mid):
    ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def evaluate(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True}, mid)
    if "exceptionDetails" in r.get("result", {}):
        print("JS_ERR:", json.dumps(r["result"]["exceptionDetails"])[:200])
    return r.get("result", {}).get("result", {}).get("value")

# Try API endpoints from inside browser
r = evaluate(ws, r"""
  (async () => {
    const projectId = 12867;
    const modelName = 'curly-crayfishwetminh-v2';
    const competitionSlug = 'datacrunch-2';

    // First GET runs list to see if API works
    const results = [];
    const bases = [
      'https://api.hub.crunchdao.com',
      'https://api.crunchdao.com',
      ''  // relative
    ];
    for (const base of bases) {
      try {
        const u = `${base}/v3/competitions/${competitionSlug}/projects/${projectId}/${modelName}/runs?submissionNumber=1`;
        const r = await fetch(u, {credentials: 'include'});
        const text = await r.text();
        results.push({url: u, status: r.status, body: text.slice(0,300)});
      } catch(e) { results.push({error: String(e).slice(0,100)}); }
    }
    return results;
  })()
""", 1)
print("LIST RUNS attempts:")
print(json.dumps(r, indent=2, ensure_ascii=False)[:3000])
ws.close()
