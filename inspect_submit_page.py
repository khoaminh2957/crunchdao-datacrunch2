"""Deep inspect CrunchDAO submit page — click Submit via CLI tab."""
import json, urllib.request, websocket, time, sys, re
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
cd = [t for t in tabs if "hub.crunchdao.com" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(cd["webSocketDebuggerUrl"], **WS_KW)

# Try numinous (likely has active round)
cdp(ws, "Page.enable", {}, 0)
cdp(ws, "Page.navigate", {"url": "https://hub.crunchdao.com/competitions/numinous/submit"}, 1)
time.sleep(8)

# Click "Submit via CLI" tab
click = evaluate(ws, r"""
  (() => {
    const all = Array.from(document.querySelectorAll('button, a, [role="tab"], [role="button"]'));
    const cli = all.find(e => /Submit via CLI|via CLI/i.test((e.innerText||'').trim()));
    if (cli) { cli.click(); return {clicked: true, text: cli.innerText}; }
    return {clicked: false, options: all.map(e => e.innerText.slice(0,40)).filter(t=>t).slice(0,20)};
  })()
""", 2)
print("CLICK CLI:", click)
time.sleep(5)

# Now scrape for setup command
result = evaluate(ws, r"""
  (() => {
    const txt = document.body.innerText;
    const matches = txt.match(/crunch setup [a-zA-Z0-9_\-]+ [a-zA-Z0-9_\-]+ --token [A-Za-z0-9]+/g) || [];
    const codes = Array.from(document.querySelectorAll('code, pre, [class*="code"]')).map(e => e.innerText.trim()).filter(t => t.includes('crunch')).slice(0,5);
    return {url: location.href, matches, codes, head: txt.slice(0,2500)};
  })()
""", 3)
print("\nRESULT:", json.dumps(result, indent=2, ensure_ascii=False)[:4000])

# Extract + save
setup_cmd = None
if result.get('matches'):
    setup_cmd = result['matches'][0]
elif result.get('codes'):
    for c in result['codes']:
        m = re.search(r'crunch setup \S+ \S+ --token \S+', c)
        if m: setup_cmd = m.group(0); break

if setup_cmd:
    print(f"\n✓ SETUP CMD: {setup_cmd}")
    with open(r"C:\Users\Admin\earn5usd\crunchdao\fresh_setup.txt", "w") as f:
        f.write(setup_cmd)
else:
    print("\n✗ Still not found")
ws.close()
