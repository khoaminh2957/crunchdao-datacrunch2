"""Cancel the in-progress submit (was bewildered), redo with curly."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if "submit/notebook" in t.get("url","") and t.get("type") == "page"][-1]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

def cdp(m, p, mid):
    ws.send(json.dumps({"id":mid,"method":m,"params":p or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def ev(expr, mid):
    r = cdp("Runtime.evaluate", {"expression":expr,"returnByValue":True,"awaitPromise":True}, mid)
    if "exceptionDetails" in r.get("result",{}):
        print("JS_ERR:", json.dumps(r["result"]["exceptionDetails"])[:300])
    return r.get("result",{}).get("result",{}).get("value")

# Click Cancel button
r = ev("""
  (() => {
    const btns = Array.from(document.querySelectorAll('button')).filter(b => b.offsetParent !== null);
    const cancel = btns.find(b => b.innerText.trim() === 'Cancel');
    if (!cancel) return {error: 'no cancel btn', avail: btns.map(b => b.innerText.trim())};
    cancel.click();
    return {clicked: true};
  })()
""", 10)
print("Cancel click:", r)
time.sleep(3)
ws.close()
print("\nClosed. Open fresh curly tab next.")
