"""Try other competitions: numinous, structural-break, synth, etc."""
import json, urllib.request, urllib.parse, websocket, time, sys, re
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
cdp(ws, "Page.enable", {}, 0)

# Try multiple competition slugs
competitions = ["numinous", "structural-break", "synth", "xentiment", "datacrunch-2", "obesity-ml-3"]
for slug in competitions:
    url = f"https://hub.crunchdao.com/competitions/{slug}/submit"
    print(f"\n=== Trying {slug} ===")
    cdp(ws, "Page.navigate", {"url": url}, 1)
    time.sleep(6)
    r = evaluate(ws, r"""
      (() => {
        const txt = document.body.innerText;
        const codes = Array.from(document.querySelectorAll('code, pre')).map(e => e.innerText.trim()).filter(t => t.includes('crunch setup'));
        const matches = txt.match(/crunch setup [a-zA-Z0-9_\-]+ [a-zA-Z0-9_\-]+ --token [A-Za-z0-9]+/g) || [];
        const has_oops = /Oops|not found|No Models Found/i.test(txt);
        const has_create_model = /Create Model/i.test(txt);
        return {url: location.href, has_oops, has_create_model, matches: matches.slice(0,1), codes_with_setup: codes.slice(0,2), head: txt.slice(0,500)};
      })()
    """, 2)
    print(f"  url: {r.get('url')[:80]}")
    print(f"  oops: {r.get('has_oops')} | needs_create_model: {r.get('has_create_model')}")
    if r.get('matches') or r.get('codes_with_setup'):
        cmd = r.get('matches', [None])[0] or r.get('codes_with_setup', [None])[0]
        m = re.search(r'crunch setup \S+ \S+ --token \S+', cmd or '')
        if m:
            print(f"  ✓ FOUND: {m.group(0)[:120]}...")
            with open(r"C:\Users\Admin\earn5usd\crunchdao\fresh_setup.txt", "w") as f:
                f.write(m.group(0))
            print(f"  Saved!")
            break
ws.close()
