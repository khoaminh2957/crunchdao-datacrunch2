"""Click 'Create' to make the new model, capture the POST API call."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tabs = [t for t in tabs if "datacrunch-2/models/create" in t.get("url","") and t.get("type") == "page"]
if not tabs:
    print("Need to open create dialog first"); sys.exit(1)
ws = websocket.create_connection(tabs[0]["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, m, p, mid):
    ws.send(json.dumps({"id":mid,"method":m,"params":p or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def ev(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression":expr,"returnByValue":True,"awaitPromise":True}, mid)
    return r.get("result",{}).get("result",{}).get("value")

cdp(ws, "Network.enable", {}, 1)

r = ev(ws, r"""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).filter(b => b.offsetParent !== null && b.innerText.trim() === 'Create');
    if (!btn[0]) return {error: 'no Create button'};
    const nameField = document.querySelector('input[name="name"]');
    const propsKey = Object.keys(btn[0]).find(k => k.startsWith('__reactProps$'));
    btn[0].click();
    return {clicked: true, name: nameField?.value};
  })()
""", 10)
print("CLICK:", r)

# Wait + capture network
events = []
ws.settimeout(0.5)
end_time = time.time() + 12
while time.time() < end_time:
    try:
        msg = json.loads(ws.recv())
        meth = msg.get("method","")
        if meth.startswith("Network.requestWillBeSent"):
            rd = msg["params"]["request"]
            u = rd.get("url","")
            if "api.hub.crunchdao.com" in u and rd.get("method") in ("POST","PUT","PATCH"):
                events.append({"REQ": rd.get("method"), "url": u, "payload": rd.get("postData","")[:300]})
        elif meth.startswith("Network.responseReceived"):
            resp = msg["params"]["response"]
            u = resp.get("url","")
            if "api.hub.crunchdao.com" in u and resp.get("status") in (200, 201):
                events.append({"RESP": u, "status": resp.get("status")})
    except websocket.WebSocketTimeoutException: pass
    except: pass
ws.settimeout(None)

print(f"\nAPI calls ({len(events)}):")
for e in events: print(" ", json.dumps(e)[:300])

# Final URL
state = ev(ws, "({url: location.href, text: document.body.innerText.slice(0, 1000)})", 99)
print(f"\nFinal URL: {state.get('url')}")
print(f"Page hint: {state.get('text','')[:500]}")
ws.close()
