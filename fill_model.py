"""Fill model name + click Create."""
import json, urllib.request, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

def cdp(ws, method, params, mid):
    ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid:
            return r

def evaluate(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate",
            {"expression": expr, "returnByValue": True, "awaitPromise": True}, mid)
    return r.get("result", {}).get("result", {}).get("value")

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
cd = [t for t in tabs if "hub.crunchdao.com" in t.get("url","") and "models/create" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(cd["webSocketDebuggerUrl"], **WS_KW)

# Focus name input + type via CDP keyboard
evaluate(ws, "document.getElementById('name').focus()", 1)
time.sleep(0.5)
evaluate(ws, "document.getElementById('name').value = ''", 1)
time.sleep(0.3)
evaluate(ws, "document.getElementById('name').focus()", 1)
time.sleep(0.3)
cdp(ws, "Input.insertText", {"text": "wetminh-v1"}, 2)
time.sleep(0.5)

# Click Create button
click = evaluate(ws, r"""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => /^Create$/i.test((b.innerText||'').trim()) && !b.disabled);
    if (btn) { btn.click(); return {clicked: true}; }
    return {clicked: false};
  })()
""", 3)
print("CREATE:", click)

# Wait for redirect to /submit
for i in range(15):
    time.sleep(1)
    url = evaluate(ws, "location.href", 10+i)
    if url and "submit" in url and "models/create" not in url:
        print(f"Redirected after {i+1}s to {url}")
        break

time.sleep(5)
# Now scrape setup command
result = evaluate(ws, r"""
  (() => {
    const txt = document.body.innerText;
    const matches = txt.match(/crunch setup [a-zA-Z0-9_\-]+ [a-zA-Z0-9_\-]+ --token [A-Za-z0-9]+/g) || [];
    const codes = Array.from(document.querySelectorAll('code, pre')).map(e => e.innerText.trim()).filter(t => t.includes('crunch setup'));
    return {
      url: location.href,
      matches_from_text: matches.slice(0,3),
      matches_from_code: codes.slice(0,3),
      head_text: txt.slice(0,2000)
    };
  })()
""", 100)
print("\nAFTER MODEL CREATE:", json.dumps(result, indent=2, ensure_ascii=False)[:3500])

# Extract command
setup_cmd = None
import re
if result.get('matches_from_code'):
    for c in result['matches_from_code']:
        m = re.search(r'crunch setup \S+ \S+ --token \S+', c)
        if m: setup_cmd = m.group(0); break
if not setup_cmd and result.get('matches_from_text'):
    setup_cmd = result['matches_from_text'][0]

if setup_cmd:
    print(f"\n✓ SETUP CMD: {setup_cmd}")
    with open(r"C:\Users\Admin\earn5usd\crunchdao\fresh_setup.txt", "w") as f:
        f.write(setup_cmd)
else:
    print("\n✗ Still no command — may need to click Submit tab")
ws.close()
