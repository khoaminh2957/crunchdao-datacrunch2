import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tabs = [t for t in tabs if "runs/create" in t.get("url","") and "submissionNumber=7" in t.get("url","") and t.get("type") == "page"]
tab = tabs[0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, m, p, mid):
    ws.send(json.dumps({"id":mid,"method":m,"params":p or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def ev(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression":expr,"returnByValue":True,"awaitPromise":True}, mid)
    if "exceptionDetails" in r.get("result",{}):
        print("JS_ERR:", r["result"]["exceptionDetails"])
    return r.get("result",{}).get("result",{}).get("value")

s = ev(ws, r"""
({
  text: document.body.innerText.slice(0, 3000),
  has_yes: !!Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Yes I did'),
  has_pw: !!document.querySelector('button[role="radio"][value="aws-cpu"]'),
  pw_state: document.querySelector('button[role="radio"][value="aws-cpu"]')?.getAttribute('aria-checked'),
  has_create: !!Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run'),
  create_disabled: Array.from(document.querySelectorAll('button')).find(b => b.innerText.trim() === 'Create run')?.disabled,
  url: location.href
})
""", 99)
print(json.dumps(s, indent=2, ensure_ascii=False)[:3000])
ws.close()
