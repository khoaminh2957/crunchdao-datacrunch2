"""Poll a run by id, arg = run id."""
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

RID = sys.argv[1]
url = f"https://hub.crunchdao.com/competitions/datacrunch-2/models/12867/curly-crayfishwetminh-v2/runs/{RID}"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(6)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if f"runs/{RID}" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

prev = None
for i in range(40):
    cdp(ws, "Page.reload", {}, 1000+i)
    time.sleep(5)
    s = evaluate(ws, r"""
      (() => {
        const txt = document.body.innerText;
        const stat = txt.match(/Status\s*\n+\s*([A-Z][a-zA-Z]+)/);
        const dur = txt.match(/Duration\s*\n+\s*([^\n]+)/);
        const scr = txt.match(/Score:\s*([\-0-9.eE]+)/);
        const val = txt.match(/Validity\s*\n+\s*([A-Za-z]+)/);
        return {status: stat?.[1], duration: dur?.[1], score: scr?.[1], validity: val?.[1], tlen: txt.length};
      })()
    """, 100+i)
    line = f"[{time.strftime('%H:%M:%S')}] {s}"
    if line != prev:
        print(line, flush=True)
        prev = line
    if s and s.get('status') in ('Success','Done','Failed','Bad','Error'):
        print(f"TERMINAL: {s}", flush=True)
        log = evaluate(ws, r"""
          (() => {
            const txt = document.body.innerText;
            const idx = Math.max(txt.lastIndexOf('Error'), txt.lastIndexOf('Traceback'), txt.lastIndexOf('Columns'));
            return idx > 0 ? txt.slice(Math.max(0, idx-200), idx+1500) : txt.slice(-2000);
          })()
        """, 999)
        print("=== LOG TAIL ===")
        print(log)
        break
    time.sleep(25)

ws.close()
