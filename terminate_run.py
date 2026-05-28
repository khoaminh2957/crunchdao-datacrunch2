"""Terminate a running run via UI Terminate button."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

RID = sys.argv[1]
PROJECT = sys.argv[2]
url = f"https://hub.crunchdao.com/competitions/datacrunch-2/models/12867/{PROJECT}/runs/{RID}"

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
for t in tabs:
    if f"runs/{RID}" in t.get("url",""):
        try: urllib.request.urlopen(f"http://localhost:9222/json/close/{t['id']}")
        except: pass
time.sleep(1)

req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(8)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if f"runs/{RID}" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, m, p, mid):
    ws.send(json.dumps({"id":mid,"method":m,"params":p or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def ev(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression":expr,"returnByValue":True,"awaitPromise":True}, mid)
    if "exceptionDetails" in r.get("result",{}):
        print("JS_ERR:", json.dumps(r["result"]["exceptionDetails"])[:300])
    return r.get("result",{}).get("result",{}).get("value")

cdp(ws, "Network.enable", {}, 1)

r = ev(ws, r"""
  (async () => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => /^Terminate/i.test(b.innerText.trim()));
    if (!btn) return {error: 'no terminate btn', buttons: Array.from(document.querySelectorAll('button')).filter(b => b.offsetParent).slice(0, 20).map(b => b.innerText.trim()).filter(t=>t)};
    btn.click();
    await new Promise(r => setTimeout(r, 1500));
    // Confirm dialog
    const confirms = Array.from(document.querySelectorAll('button')).filter(b => b.offsetParent && /Terminate|Confirm|Yes|OK/i.test(b.innerText.trim()));
    console.log('confirms:', confirms.length);
    if (confirms.length > 0) {
      // click the last/strongest one
      confirms[confirms.length - 1].click();
    }
    return {clicked: true, confirms_found: confirms.length, confirm_texts: confirms.map(c => c.innerText.trim())};
  })()
""", 10)
print("TERMINATE:", r)

# Capture POST/DELETE/PATCH for the terminate API
events = []
ws.settimeout(0.5)
end_time = time.time() + 10
while time.time() < end_time:
    try:
        msg = json.loads(ws.recv())
        if msg.get("method","").startswith("Network.requestWillBeSent"):
            rd = msg["params"]["request"]
            u = rd.get("url","")
            if "api.hub.crunchdao.com" in u and rd.get("method") in ("POST","PUT","PATCH","DELETE"):
                events.append({"REQ": rd.get("method"), "url": u, "payload": rd.get("postData","")[:200]})
    except websocket.WebSocketTimeoutException: pass
    except: pass
ws.settimeout(None)

for e in events: print(" ", e)

state = ev(ws, "({url: location.href, text: document.body.innerText.slice(0,1500)})", 99)
print(f"\nFinal URL: {state.get('url')}")
print("Hint:", state.get('text','')[:500])
ws.close()
