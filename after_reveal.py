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

# Get ALL text including hidden, ARIA labels, data attrs
result = evaluate(ws, r"""
  (() => {
    const html = document.documentElement.outerHTML;
    // Find any --token in raw HTML (might be in attr/data)
    const tokens = html.match(/--token\s+[A-Za-z0-9]{20,}/g) || [];
    const tokensInAttr = html.match(/token['"]\s*[:=]\s*['"][A-Za-z0-9]{50,}['"]/g) || [];
    // Find all inputs/textareas with values
    const inputs = Array.from(document.querySelectorAll('input, textarea')).map(i => ({
      type: i.type, name: i.name, id: i.id, value: i.value.slice(0,200), placeholder: i.placeholder
    })).filter(i => i.value);
    return {
      url: location.href,
      html_len: html.length,
      tokens_in_html: tokens.slice(0,3),
      tokens_in_attrs: tokensInAttr.slice(0,3),
      input_values: inputs.slice(0,10),
      // Re-grep visible text
      visible_text: document.body.innerText.slice(0,3500)
    };
  })()
""", 1)
print(json.dumps(result, indent=2, ensure_ascii=False)[:5000])

# Save HTML for inspection
html = evaluate(ws, "document.documentElement.outerHTML", 2)
with open(r"C:\Users\Admin\earn5usd\crunchdao\after_reveal.html", "w", encoding="utf-8") as f:
    f.write(html or "")
print(f"\nHTML saved {len(html or '')} chars")
ws.close()
