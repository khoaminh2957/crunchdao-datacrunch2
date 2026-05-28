"""See submissions table + check if a select-for-final action exists."""
import json, urllib.request, urllib.parse, websocket, time, sys, base64
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

PROJECT = sys.argv[1] if len(sys.argv) > 1 else "curly-crayfishwetminh-v2"
url = f"https://hub.crunchdao.com/competitions/datacrunch-2/projects/12867/{PROJECT}/submissions/10"

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
for t in tabs:
    if f"/{PROJECT}/submissions" in t.get("url",""):
        try: urllib.request.urlopen(f"http://localhost:9222/json/close/{t['id']}")
        except: pass
time.sleep(1)

req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(8)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if f"/{PROJECT}/submissions" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, m, p, mid):
    ws.send(json.dumps({"id":mid,"method":m,"params":p or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def ev(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression":expr,"returnByValue":True,"awaitPromise":True}, mid)
    return r.get("result",{}).get("result",{}).get("value")

s = ev(ws, r"""
({
  text: document.body.innerText.slice(0, 8000),
  hasSelect: !!Array.from(document.querySelectorAll('button')).find(b => /select/i.test(b.innerText)),
  buttons: Array.from(document.querySelectorAll('button')).filter(b => b.offsetParent).slice(0,40).map(b => b.innerText.trim()).filter(t => t && t.length < 40)
})
""", 50)
print("BUTTONS sample:", s.get('buttons',[]))
print("\n=== PAGE TEXT ===")
print(s.get('text',''))

ss = cdp(ws, "Page.captureScreenshot", {"format":"png","captureBeyondViewport":True}, 51)
img = ss.get("result",{}).get("data","")
if img:
    open(f"C:/Users/Admin/earn5usd/crunchdao/submissions_{PROJECT}.png","wb").write(base64.b64decode(img))
    print(f"\nScreenshot -> submissions_{PROJECT}.png")
ws.close()
