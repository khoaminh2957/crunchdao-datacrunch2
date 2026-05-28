"""Fetch the 'Submit via CLI' setup command (contains clone token) for bewildered-chickadee."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

PROJECT = sys.argv[1] if len(sys.argv) > 1 else "bewildered-chickadee"
url = f"https://hub.crunchdao.com/competitions/datacrunch-2/submit/cli?projectName={PROJECT}"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(8)

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
    return r.get("result",{}).get("result",{}).get("value")

# Click 'Reveal' if present
ev(ws, r"""
  (() => {
    const r = Array.from(document.querySelectorAll('button')).find(b => /^Reveal/i.test(b.innerText.trim()));
    if (r) r.click();
  })()
""", 5)
time.sleep(2)

state = ev(ws, r"""
({
  url: location.href,
  text: document.body.innerText,
  // Find any code-like block that contains 'crunch setup'
  setup_commands: Array.from(document.querySelectorAll('code, pre')).map(e => e.innerText).filter(t => /crunch setup/.test(t))
})
""", 10)
print(f"URL: {state.get('url')}")
print(f"\n=== Setup commands found ({len(state.get('setup_commands',[]))}):")
for c in state.get('setup_commands', []):
    print("---")
    print(c)
print("\n=== Page text snippet ===")
txt = state.get('text','')
idx = txt.lower().find('crunch setup')
if idx >= 0:
    print(txt[max(0,idx-100):idx+800])
ws.close()
