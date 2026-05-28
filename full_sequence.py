"""Full sequence: Getting started → I'm Ready → Create model → Get setup command."""
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

# Fresh navigation
cdp(ws, "Page.enable", {}, 0)
cdp(ws, "Page.navigate", {"url": "https://hub.crunchdao.com/competitions/datacrunch-2/submit"}, 1)
time.sleep(10)

# Step 1: click Getting started
print("=== Step 1: Click Getting started ===")
r1 = evaluate(ws, r"""
  (() => {
    const btns = Array.from(document.querySelectorAll('button'));
    const gs = btns.find(b => /Getting started/i.test((b.innerText||'').trim()));
    if (gs) { gs.click(); return {clicked: 'Getting started'}; }
    return {clicked: null};
  })()
""", 2)
print(r1)
time.sleep(3)

# Step 2: in dialog, click I'm Ready to Compete
print("\n=== Step 2: Click I'm Ready to Compete ===")
r2 = evaluate(ws, r"""
  (() => {
    // Look INSIDE dialog
    const dialog = document.querySelector('[role="dialog"]');
    const scope = dialog || document;
    const btns = Array.from(scope.querySelectorAll('button'));
    const ready = btns.find(b => /Ready to Compete/i.test((b.innerText||'').trim()));
    if (ready) { ready.click(); return {clicked: 'Ready', inDialog: !!dialog}; }
    return {clicked: null, dialogPresent: !!dialog, options: btns.map(b => b.innerText.slice(0,40)).filter(t=>t).slice(0,10)};
  })()
""", 3)
print(r2)
time.sleep(5)

# Step 3: check if model creation needed
print("\n=== Step 3: State check ===")
r3 = evaluate(ws, r"""
  (() => {
    const txt = document.body.innerText;
    const matches = txt.match(/crunch setup [a-zA-Z0-9_\-]+ [a-zA-Z0-9_\-]+ --token [A-Za-z0-9]+/g) || [];
    const codes = Array.from(document.querySelectorAll('code, pre')).map(e => e.innerText.trim()).filter(t => t.includes('crunch')).slice(0,3);
    const has_create_model = /create model|new model|No Models Found/i.test(txt);
    return {url: location.href, matches, codes, has_create_model, head: txt.slice(0,1000)};
  })()
""", 4)
print(json.dumps(r3, indent=2, ensure_ascii=False)[:2500])

# Step 4: if need create model, do it
if r3.get('has_create_model') and not (r3.get('matches') or r3.get('codes')):
    print("\n=== Step 4: Create Model ===")
    r4 = evaluate(ws, r"""
      (() => {
        const all = Array.from(document.querySelectorAll('button, a'));
        const btn = all.find(b => /create model|new model/i.test((b.innerText||'').trim()));
        if (btn) { btn.click(); return {clicked: btn.innerText, href: btn.href}; }
        return {clicked: null};
      })()
    """, 5)
    print(r4)
    time.sleep(5)
    # Fill name
    evaluate(ws, "document.getElementById('name')?.focus()", 6)
    time.sleep(0.5)
    cdp(ws, "Input.insertText", {"text": "wetminh-v2"}, 7)
    time.sleep(0.5)
    r5 = evaluate(ws, r"""
      (() => {
        const btn = Array.from(document.querySelectorAll('button')).find(b => /^Create$/i.test((b.innerText||'').trim()) && !b.disabled);
        if (btn) { btn.click(); return {clicked: true}; }
        return {clicked: false};
      })()
    """, 8)
    print("Create submit:", r5)
    time.sleep(8)

# Final check
print("\n=== FINAL ===")
final = evaluate(ws, r"""
  (() => {
    const txt = document.body.innerText;
    const matches = txt.match(/crunch setup [a-zA-Z0-9_\-]+ [a-zA-Z0-9_\-]+ --token [A-Za-z0-9]+/g) || [];
    const codes = Array.from(document.querySelectorAll('code, pre')).map(e => e.innerText.trim()).filter(t => t.includes('crunch')).slice(0,3);
    return {url: location.href, matches, codes, head: txt.slice(0,1500)};
  })()
""", 100)
print(json.dumps(final, indent=2, ensure_ascii=False)[:3000])

if final.get('matches') or final.get('codes'):
    cmd = final.get('matches', [None])[0]
    if not cmd:
        for c in final.get('codes', []):
            m = re.search(r'crunch setup \S+ \S+ --token \S+', c)
            if m: cmd = m.group(0); break
    if cmd:
        print(f"\n✓✓✓ FOUND: {cmd}")
        with open(r"C:\Users\Admin\earn5usd\crunchdao\fresh_setup.txt", "w") as f:
            f.write(cmd)
        print("Saved to fresh_setup.txt")
ws.close()
