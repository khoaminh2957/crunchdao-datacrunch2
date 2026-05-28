"""Trigger cloud run via direct API call inside authed CDP browser."""
import json, urllib.request, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)
tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
run_tab = [t for t in tabs if "runs/create" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(run_tab["webSocketDebuggerUrl"], **WS_KW)

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

# Try multiple endpoint patterns
result = evaluate(ws, r"""
  (async () => {
    const projectId = 12867;
    const modelName = 'curly-crayfishwetminh-v2';
    const competitionSlug = 'datacrunch-2';
    const submissionNumber = 1;

    // Body shape from form fields
    const body = {
      submissionNumber: submissionNumber,
      instanceType: 'aws-cpu',
      forceFirstTrain: false,
      trainFrequency: 0,
      localTested: true
    };

    const endpoints = [
      `/api/v1/competitions/${competitionSlug}/projects/${projectId}/${modelName}/runs`,
      `/api/v1/competitions/${competitionSlug}/models/${projectId}/${modelName}/runs`,
      `/api/competitions/${competitionSlug}/projects/${projectId}/runs`,
      `/api/v1/projects/${projectId}/runs`,
      `https://api.crunchdao.com/v1/competitions/${competitionSlug}/projects/${projectId}/runs`,
      `https://api.hub.crunchdao.com/v1/competitions/${competitionSlug}/projects/${projectId}/runs`,
    ];
    const results = [];
    for (const url of endpoints) {
      try {
        const r = await fetch(url, {
          method: 'POST',
          credentials: 'include',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body)
        });
        const text = await r.text();
        results.push({url, status: r.status, body: text.slice(0,300)});
        if (r.status < 400) return {success: url, status: r.status, body: text.slice(0,500)};
      } catch (e) {
        results.push({url, error: String(e).slice(0,100)});
      }
    }
    return {tried_all: results};
  })()
""", 1)
print("API RESULT:", json.dumps(result, indent=2, ensure_ascii=False)[:5000])
ws.close()
