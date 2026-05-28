import json, urllib.request, urllib.parse, websocket, time, sys, re
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

url = "https://hub.crunchdao.com/competitions/datacrunch-2/submit/cli?projectName=historic-skunk"

# Close older tabs
tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
for t in tabs:
    if "submit/cli" in t.get("url","") and "historic-skunk" in t.get("url",""):
        try: urllib.request.urlopen(f"http://localhost:9222/json/close/{t['id']}")
        except: pass
time.sleep(1)
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(10)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if "submit/cli" in t.get("url","") and "historic-skunk" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, m, p, mid):
    ws.send(json.dumps({"id":mid,"method":m,"params":p or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def ev(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression":expr,"returnByValue":True,"awaitPromise":True}, mid)
    return r.get("result",{}).get("result",{}).get("value")

# Click Reveal
ev(ws, """
  (() => {
    const r = Array.from(document.querySelectorAll('button')).filter(b => /Reveal/i.test(b.innerText));
    r.forEach(x=>x.click());
    return {clicked: r.length};
  })()
""", 5)
time.sleep(3)

txt = ev(ws, "document.body.innerText", 10)
m = re.search(r"crunch setup\s+datacrunch-2\s+historic-skunk\s+--token\s+(\S+)", txt or "")
if m:
    print(f"TOKEN: {m.group(1)}")
else:
    idx = (txt or "").find("crunch setup")
    if idx > 0:
        print(txt[idx:idx+600])
    else:
        print("NO SETUP FOUND. Snippet:")
        print((txt or "")[:1500])
ws.close()
