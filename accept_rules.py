"""Accept rules on CrunchDAO competition + find setup command."""
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

# Navigate to datacrunch-2 (active competition)
cdp(ws, "Page.enable", {}, 0)
cdp(ws, "Page.navigate", {"url": "https://hub.crunchdao.com/competitions/datacrunch-2/submit"}, 1)
time.sleep(7)

# Click through modals + accept rules (skip Close, prefer affirmative actions)
for round_n in range(5):
    print(f"\n=== Round {round_n+1} ===")
    actions = evaluate(ws, r"""
      (() => {
        const all = Array.from(document.querySelectorAll('button, a, [role="button"]'));
        const buttons = all.map(e => ({text: (e.innerText||'').trim().slice(0,60), tag: e.tagName, href: e.href || null}));
        // Click any of: Close (modal), Next, Getting started, Accept rules, I agree, Continue
        const targets = ['^Next$', '^Getting started', '^Accept', '^I agree', '^I accept', '^Continue', '^I have read', '^Confirm'];
        for (const pattern of targets) {
          const btn = all.find(b => new RegExp(pattern, 'i').test((b.innerText||'').trim()) && !b.disabled);
          if (btn) {
            btn.click();
            return {clicked: btn.innerText.trim().slice(0,40), pattern};
          }
        }
        return {clicked: null, available: buttons.slice(0,15)};
      })()
    """, 1 + round_n*10)
    print("  Action:", actions)
    time.sleep(3)

    # Check if we have setup command yet
    result = evaluate(ws, r"""
      (() => {
        const txt = document.body.innerText;
        const matches = txt.match(/crunch setup [a-zA-Z0-9_\-]+ [a-zA-Z0-9_\-]+ --token [A-Za-z0-9]+/g) || [];
        const codes = Array.from(document.querySelectorAll('code, pre')).map(e => e.innerText.trim()).filter(t => t.includes('crunch setup')).slice(0,2);
        return {url: location.href, matches, codes, text_head: txt.slice(0,500)};
      })()
    """, 2 + round_n*10)
    if result.get('matches') or result.get('codes'):
        setup_cmd = None
        if result.get('matches'): setup_cmd = result['matches'][0]
        else:
            for c in result['codes']:
                m = re.search(r'crunch setup \S+ \S+ --token \S+', c)
                if m: setup_cmd = m.group(0); break
        if setup_cmd:
            print(f"\n✓ FOUND: {setup_cmd}")
            with open(r"C:\Users\Admin\earn5usd\crunchdao\fresh_setup.txt", "w") as f:
                f.write(setup_cmd)
            break
    if not actions.get('clicked'):
        print("  No more buttons to click, stopping")
        print(f"  Page head: {result.get('text_head','')[:400]}")
        break
ws.close()
