"""Click 'Submit via CLI' tab to reveal setup command."""
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
    if "exceptionDetails" in r.get("result", {}):
        print("JS_ERR:", json.dumps(r["result"]["exceptionDetails"])[:200])
    return r.get("result", {}).get("result", {}).get("value")

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
cd = [t for t in tabs if "hub.crunchdao.com" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(cd["webSocketDebuggerUrl"], **WS_KW)

# Click Submit via CLI tab
click = evaluate(ws, r"""
  (() => {
    const all = Array.from(document.querySelectorAll('button, a, [role="tab"]'));
    const tab = all.find(e => /Submit via CLI/i.test((e.innerText||'').trim()));
    if (tab) { tab.click(); return {clicked: true, text: tab.innerText, href: tab.href}; }
    return {clicked: false, tabs: all.map(e => e.innerText.slice(0,30)).filter(t=>t).slice(0,20)};
  })()
""", 1)
print("CLICK CLI:", click)
time.sleep(6)

# Scrape command
result = evaluate(ws, r"""
  (() => {
    const txt = document.body.innerText;
    const matches = txt.match(/crunch setup [a-zA-Z0-9_\-]+ [a-zA-Z0-9_\-]+ --token [A-Za-z0-9]+/g) || [];
    const codes = Array.from(document.querySelectorAll('code, pre, [class*="code"]')).map(e => e.innerText.trim()).filter(t => t.includes('crunch')).slice(0,5);
    return {url: location.href, matches, codes, head: txt.slice(0,2500)};
  })()
""", 2)
print("\nFINAL:", json.dumps(result, indent=2, ensure_ascii=False)[:4500])

if result.get('matches') or result.get('codes'):
    cmd = result.get('matches', [None])[0]
    if not cmd:
        for c in result.get('codes', []):
            m = re.search(r'crunch setup \S+ \S+ --token \S+', c)
            if m: cmd = m.group(0); break
    if cmd:
        print(f"\n✓ FOUND: {cmd}")
        with open(r"C:\Users\Admin\earn5usd\crunchdao\fresh_setup.txt", "w") as f:
            f.write(cmd)
ws.close()
