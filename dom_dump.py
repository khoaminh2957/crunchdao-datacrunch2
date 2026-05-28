"""Aggressive DOM dump + grep for setup command anywhere."""
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
cdp(ws, "Page.enable", {}, 0)

# Try datacrunch-2 first since wetminh-v1 model created there
cdp(ws, "Page.navigate", {"url": "https://hub.crunchdao.com/competitions/datacrunch-2/submit"}, 1)
time.sleep(10)

# Aggressive: close ALL modals + dismiss popups
for _ in range(3):
    evaluate(ws, r"""
      (() => {
        // ESC key + click off-canvas
        document.body.click();
        document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', code: 'Escape'}));
        // Close any visible modals by clicking close X
        const xs = Array.from(document.querySelectorAll('button[aria-label*="close" i], button[aria-label*="dismiss" i]'));
        xs.forEach(b => b.click());
      })()
    """, 5)
    time.sleep(2)

# Wait more for full render
time.sleep(5)

# Dump COMPLETE HTML to file
html = evaluate(ws, "document.documentElement.outerHTML", 10)
if html:
    with open(r"C:\Users\Admin\earn5usd\crunchdao\dom_dump.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"DOM dumped: {len(html)} chars")

    # Grep for setup-related patterns
    patterns = [
        r"crunch setup [^<\"'\s]+",
        r"--token [A-Za-z0-9]+",
        r"setupCommand['\"]\s*[:=]\s*['\"][^'\"]+",
        r"setupToken['\"]\s*[:=]\s*['\"][^'\"]+",
        r"\"token\"\s*:\s*\"[A-Za-z0-9]{50,}\"",
        r"data-token=['\"][A-Za-z0-9]{50,}['\"]"
    ]
    for p in patterns:
        matches = re.findall(p, html)
        if matches:
            print(f"\nPattern '{p[:40]}': {len(matches)} matches")
            for m in matches[:3]:
                print(f"  -> {m[:200]}")

# Also get visible page text
txt = evaluate(ws, "document.body.innerText", 11)
print(f"\nVisible text ({len(txt or '')} chars):")
print((txt or '')[:3000])
ws.close()
