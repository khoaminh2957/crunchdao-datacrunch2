"""Click 'I'm Ready to Compete!' + extract setup command."""
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

# Click "I'm Ready to Compete!"
click = evaluate(ws, r"""
  (() => {
    const all = Array.from(document.querySelectorAll('button, a, [role="button"]'));
    const btn = all.find(b => /Ready to Compete/i.test((b.innerText||'').trim()));
    if (btn) { btn.click(); return {clicked: true, text: btn.innerText}; }
    return {clicked: false, available: all.map(e => e.innerText.slice(0,50)).filter(t=>t).slice(0,20)};
  })()
""", 1)
print("READY:", click)
time.sleep(7)

# Check state — should now show submit page with model creation
state = evaluate(ws, r"""
  (() => {
    const txt = document.body.innerText;
    const matches = txt.match(/crunch setup [a-zA-Z0-9_\-]+ [a-zA-Z0-9_\-]+ --token [A-Za-z0-9]+/g) || [];
    const codes = Array.from(document.querySelectorAll('code, pre, [class*="code"]')).map(e => e.innerText.trim()).filter(t => t.includes('crunch')).slice(0,3);
    return {
      url: location.href,
      matches, codes,
      head: txt.slice(0,2500),
      // Look for create model button
      has_create_model: /create model|new model/i.test(txt)
    };
  })()
""", 2)
print("\nSTATE:", json.dumps(state, indent=2, ensure_ascii=False)[:4000])

# If setup command found, save + activate
if state.get('matches') or state.get('codes'):
    cmd = state.get('matches', [None])[0]
    if not cmd:
        for c in state.get('codes', []):
            m = re.search(r'crunch setup \S+ \S+ --token \S+', c)
            if m: cmd = m.group(0); break
    if cmd:
        print(f"\n✓ FOUND: {cmd}")
        with open(r"C:\Users\Admin\earn5usd\crunchdao\fresh_setup.txt", "w") as f:
            f.write(cmd)
elif state.get('has_create_model'):
    print("\n→ Now need to create model (click 'Create Model')")
    click2 = evaluate(ws, r"""
      (() => {
        const all = Array.from(document.querySelectorAll('button, a, [role="button"]'));
        const btn = all.find(b => /create model|new model/i.test((b.innerText||'').trim()));
        if (btn) { btn.click(); return {clicked: true, href: btn.href}; }
        return {clicked: false};
      })()
    """, 3)
    print("Create model click:", click2)
    time.sleep(6)
    # Fill form
    evaluate(ws, "document.getElementById('name')?.focus()", 4)
    time.sleep(0.5)
    cdp(ws, "Input.insertText", {"text": "wetminh-v1"}, 5)
    time.sleep(0.5)
    submit_create = evaluate(ws, r"""
      (() => {
        const btn = Array.from(document.querySelectorAll('button')).find(b => /^Create$/i.test((b.innerText||'').trim()) && !b.disabled);
        if (btn) { btn.click(); return {clicked: true}; }
        return {clicked: false};
      })()
    """, 6)
    print("Submit create:", submit_create)
    time.sleep(8)
    # Check final
    final = evaluate(ws, r"""
      (() => {
        const txt = document.body.innerText;
        const matches = txt.match(/crunch setup [a-zA-Z0-9_\-]+ [a-zA-Z0-9_\-]+ --token [A-Za-z0-9]+/g) || [];
        const codes = Array.from(document.querySelectorAll('code, pre')).map(e => e.innerText.trim()).filter(t => t.includes('crunch')).slice(0,3);
        return {url: location.href, matches, codes, head: txt.slice(0,1500)};
      })()
    """, 100)
    print("\nFINAL:", json.dumps(final, indent=2, ensure_ascii=False)[:3000])
    if final.get('matches') or final.get('codes'):
        cmd = final.get('matches', [None])[0]
        if not cmd:
            for c in final.get('codes', []):
                m = re.search(r'crunch setup \S+ \S+ --token \S+', c)
                if m: cmd = m.group(0); break
        if cmd:
            print(f"\n✓ FOUND AFTER CREATE: {cmd}")
            with open(r"C:\Users\Admin\earn5usd\crunchdao\fresh_setup.txt", "w") as f:
                f.write(cmd)
ws.close()
