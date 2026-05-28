"""Open CrunchDAO hub setup page via CDP, scrape fresh setup command, execute it."""
import json, urllib.request, urllib.parse, websocket, time, sys, subprocess, re
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

def open_tab(url):
    req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
    return json.loads(urllib.request.urlopen(req).read())

# Step 1: open hub setup page
print("Opening hub.crunchdao.com setup page...")
tab_meta = open_tab("https://hub.crunchdao.com/competitions/datacrunch/submit")
time.sleep(10)

# Step 2: find CrunchDAO tab
tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
cd = [t for t in tabs if "crunchdao.com" in t.get("url","") and "competition" in t.get("url","") and t.get("type") == "page"]
if not cd:
    print("NO_CRUNCHDAO_TAB"); sys.exit(1)
tab = cd[0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)
print(f"Tab: {tab['url']}")

# Wait for SPA render
for i in range(30):
    time.sleep(1)
    txt_len = evaluate(ws, "document.body.innerText.length", 1+i)
    if isinstance(txt_len, int) and txt_len > 1000:
        print(f"Loaded after {i+1}s ({txt_len} chars)")
        break

# Step 3: find setup command in page
# Common pattern: <code>crunch setup datacrunch ... --token ...</code>
result = evaluate(ws, r"""
  (() => {
    const txt = document.body.innerText;
    // Find "crunch setup" lines
    const matches = txt.match(/crunch setup [a-zA-Z0-9_\-]+ [a-zA-Z0-9_\-]+ --token [A-Za-z0-9]+/g) || [];
    // Also check <code> blocks
    const codes = Array.from(document.querySelectorAll('code, pre')).map(e => e.innerText).filter(t => t.includes('crunch setup'));
    return {
      matches_from_text: matches.slice(0,3),
      matches_from_code: codes.slice(0,3),
      head_text: txt.slice(0,2000)
    };
  })()
""", 100)
print(json.dumps(result, indent=2, ensure_ascii=False)[:3500])

# Try to extract setup command
setup_cmd = None
if result and result.get('matches_from_code'):
    for c in result['matches_from_code']:
        m = re.search(r'crunch setup \S+ \S+ --token \S+', c)
        if m:
            setup_cmd = m.group(0)
            break
if not setup_cmd and result and result.get('matches_from_text'):
    setup_cmd = result['matches_from_text'][0]

if setup_cmd:
    print(f"\n✓ FOUND SETUP COMMAND:")
    print(setup_cmd)
    # Save for later use
    with open(r"C:\Users\Admin\earn5usd\crunchdao\fresh_setup.txt", "w") as f:
        f.write(setup_cmd)
    print(f"\nSaved to fresh_setup.txt")
else:
    print("\n✗ NO SETUP COMMAND FOUND IN PAGE")
    print("Need to find another way - may need to click 'Reveal Token' button or similar")
ws.close()
