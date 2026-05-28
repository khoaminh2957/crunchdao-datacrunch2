"""Poll a run by id + project, arg1=run_id, arg2=project."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

def cdp(ws, m, p, mid):
    ws.send(json.dumps({"id":mid,"method":m,"params":p or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def ev(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression":expr,"returnByValue":True,"awaitPromise":True}, mid)
    return r.get("result",{}).get("result",{}).get("value")

RID = sys.argv[1]
PROJECT = sys.argv[2]
url = f"https://hub.crunchdao.com/competitions/datacrunch-2/models/12867/{PROJECT}/runs/{RID}"

# Close existing tab
tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
for t in tabs:
    if f"runs/{RID}" in t.get("url",""):
        try: urllib.request.urlopen(f"http://localhost:9222/json/close/{t['id']}")
        except: pass
time.sleep(1)

req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(6)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if f"runs/{RID}" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

prev = None
for i in range(60):
    cdp(ws, "Page.reload", {}, 1000+i)
    time.sleep(5)
    s = ev(ws, r"""
      (() => {
        const txt = document.body.innerText;
        const stat = txt.match(/Status\s*\n+\s*([A-Z][a-zA-Z]+)/);
        const dur = txt.match(/Duration\s*\n+\s*([^\n]+)/);
        const scr = txt.match(/Score:\s*([\-0-9.eE]+)/);
        return {status: stat?.[1], duration: dur?.[1], score: scr?.[1], tlen: txt.length};
      })()
    """, 100+i)
    line = f"[{time.strftime('%H:%M:%S')}] {s}"
    if line != prev:
        print(line, flush=True); prev = line
    if s and s.get('status') in ('Success','Done','Failed','Bad','Error'):
        print(f"TERMINAL {RID} ({PROJECT}): {s}", flush=True)
        break
    time.sleep(25)

ws.close()
