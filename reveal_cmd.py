"""Click Reveal button to show setup command."""
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
cd = [t for t in tabs if "hub.crunchdao.com" in t.get("url","") and "/submit/cli" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(cd["webSocketDebuggerUrl"], **WS_KW)

# Find Reveal button near "Setup commands"
click = evaluate(ws, r"""
  (() => {
    const all = Array.from(document.querySelectorAll('button'));
    // Reveal near Setup commands section
    const reveal = all.find(b => /^Reveal/i.test((b.innerText||'').trim()));
    if (reveal) { reveal.click(); return {clicked: 'Reveal'}; }
    return {clicked: null, btns: all.map(b => b.innerText.slice(0,30)).filter(t=>t).slice(0,20)};
  })()
""", 1)
print("CLICK:", click)
time.sleep(5)

# Scrape
result = evaluate(ws, r"""
  (() => {
    const txt = document.body.innerText;
    const matches = txt.match(/crunch setup [a-zA-Z0-9_\-]+ [a-zA-Z0-9_\-]+ --token [A-Za-z0-9]+/g) || [];
    const codes = Array.from(document.querySelectorAll('code, pre, [class*="code"]')).map(e => e.innerText.trim()).filter(t => t.includes('crunch')).slice(0,5);
    return {url: location.href, matches, codes, all_text_with_setup: txt.split('\n').filter(l => l.includes('crunch') || l.includes('--token')).slice(0,10)};
  })()
""", 2)
print("\nRESULT:", json.dumps(result, indent=2, ensure_ascii=False)[:4000])

if result.get('matches') or result.get('codes'):
    cmd = result.get('matches', [None])[0]
    if not cmd:
        for c in result.get('codes', []):
            m = re.search(r'crunch setup \S+ \S+ --token \S+', c)
            if m: cmd = m.group(0); break
    if cmd:
        print(f"\n✓✓✓ FOUND: {cmd}")
        with open(r"C:\Users\Admin\earn5usd\crunchdao\fresh_setup.txt", "w") as f:
            f.write(cmd)
ws.close()
