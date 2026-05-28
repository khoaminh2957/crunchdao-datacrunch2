"""Create project via dropdown click (the way that worked before)."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

# Open submit page on existing project (any), then click model dropdown
url = "https://hub.crunchdao.com/competitions/datacrunch-2/submit/notebook"
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(10)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
# Get the newest submit/notebook tab
matching = sorted([t for t in tabs if "submit/notebook" in t.get("url","") and t.get("type") == "page"], key=lambda t: t.get('id'))
tab = matching[-1]
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

# 1. Click model dropdown
r = ev(ws, """
  (async () => {
    // Find buttons that match any of our project names
    const projects = ['curly-crayfishwetminh-v2', 'bewildered-chickadee', 'historic-skunk', 'efficient-anteater'];
    const triggers = Array.from(document.querySelectorAll('button')).filter(b => {
      const t = b.innerText.trim();
      return projects.includes(t) && b.offsetParent !== null;
    });
    if (triggers.length === 0) return {error: 'no dropdown trigger'};
    triggers[0].click();
    await new Promise(r => setTimeout(r, 1000));
    return {opened: triggers.length, names: triggers.map(t => t.innerText.trim())};
  })()
""", 10)
print("DROPDOWN:", r)
time.sleep(2)

# 2. Find New model item
r = ev(ws, """
  (async () => {
    const items = Array.from(document.querySelectorAll('button, [role="menuitem"], [role="option"], div'));
    const nm = items.find(el => /^New model/i.test(el.innerText.trim()) && el.offsetParent !== null);
    if (!nm) return {error: 'no New model item', sample: items.filter(e => e.offsetParent).slice(0,5).map(e => e.innerText.trim().slice(0,30))};
    nm.click();
    return {clicked_new_model: true};
  })()
""", 11)
print("NEW MODEL:", r)
time.sleep(5)

state = ev(ws, "({url: location.href})", 50)
print(f"After URL: {state.get('url')}")

# If we landed on models/create, click Create
if state and "/models/create" in state.get('url',''):
    r = ev(ws, """
      (() => {
        const nameField = document.querySelector('input[name="name"]');
        const btn = Array.from(document.querySelectorAll('button')).filter(b => b.offsetParent !== null && b.innerText.trim() === 'Create')[0];
        if (!btn) return {error: 'no Create btn'};
        const name = nameField?.value;
        btn.click();
        return {clicked: true, name};
      })()
    """, 20)
    print("CREATE:", r)
    time.sleep(4)
    state = ev(ws, "({url: location.href})", 51)
    print(f"Final URL: {state.get('url')}")

ws.close()
