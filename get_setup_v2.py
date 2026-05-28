"""Better setup token grabber — look across full DOM."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

PROJECT = sys.argv[1] if len(sys.argv) > 1 else "bewildered-chickadee"
url = f"https://hub.crunchdao.com/competitions/datacrunch-2/submit/cli?projectName={PROJECT}"

# Close existing
tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
for t in tabs:
    if "submit/cli" in t.get("url","") and PROJECT in t.get("url",""):
        try: urllib.request.urlopen(f"http://localhost:9222/json/close/{t['id']}")
        except: pass
time.sleep(1)

req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(10)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tabs = [t for t in tabs if "submit/cli" in t.get("url","") and PROJECT in t.get("url","") and t.get("type") == "page"]
ws = websocket.create_connection(tabs[0]["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, m, p, mid):
    ws.send(json.dumps({"id":mid,"method":m,"params":p or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def ev(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression":expr,"returnByValue":True,"awaitPromise":True}, mid)
    if "exceptionDetails" in r.get("result",{}):
        print("JS_ERR:", json.dumps(r["result"]["exceptionDetails"])[:200])
    return r.get("result",{}).get("result",{}).get("value")

# Click 'Reveal' (multiple buttons may exist)
clicked = ev(ws, r"""
  (() => {
    const btns = Array.from(document.querySelectorAll('button')).filter(b => /Reveal/i.test(b.innerText));
    btns.forEach(b => b.click());
    return {clicked: btns.length};
  })()
""", 5)
print("REVEAL:", clicked)
time.sleep(3)

# Look for setup command containing token
state = ev(ws, r"""
({
  text: document.body.innerText,
  url: location.href,
  // All visible code-like blocks
  codes: Array.from(document.querySelectorAll('code, pre, [class*="code"]')).map(e => e.innerText).filter(t => t && t.length > 10).slice(0, 20)
})
""", 10)
txt = state.get('text','')
import re
# Match crunch setup commands
patterns = [
    r'crunch setup[^\n]+',
    r'python -m crunch setup[^\n]+',
]
for p in patterns:
    for m in re.finditer(p, txt):
        print(f"\n>>> {m.group(0)}")

print("\n=== code blocks ===")
for c in state.get('codes', []):
    if 'setup' in c.lower() or 'token' in c.lower():
        print("---"); print(c)

ws.close()
