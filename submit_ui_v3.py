"""Find React onClick of submit button, invoke directly."""
import json, urllib.request, urllib.parse, websocket, time, sys, os
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

PROJECT = "bewildered-chickadee"
NOTEBOOK = r"C:\Users\Admin\earn5usd\crunchdao\v60_submission_files\v60.ipynb"

# Find existing tab
tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if "submit/notebook" in t.get("url","") and PROJECT in t.get("url","") and t.get("type") == "page"][-1]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

def cdp(m, p, mid):
    ws.send(json.dumps({"id":mid,"method":m,"params":p or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def ev(expr, mid):
    r = cdp("Runtime.evaluate", {"expression":expr,"returnByValue":True,"awaitPromise":True}, mid)
    if "exceptionDetails" in r.get("result",{}):
        print("JS_ERR:", json.dumps(r["result"]["exceptionDetails"])[:500])
    return r.get("result",{}).get("result",{}).get("value")

cdp("Network.enable", {}, 1)

# Inspect the submit button's React structure
info = ev("""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => /Submit to/.test(b.innerText));
    if (!btn) return {error: 'no btn'};
    const keys = Object.keys(btn);
    const propsKey = keys.find(k => k.startsWith('__reactProps$'));
    const fiberKey = keys.find(k => k.startsWith('__reactFiber$'));
    const props = propsKey ? btn[propsKey] : {};
    const fiber = fiberKey ? btn[fiberKey] : null;
    let ancestor = fiber;
    let formNodes = [];
    while (ancestor && formNodes.length < 8) {
      if (ancestor.stateNode && ancestor.stateNode.tagName === 'FORM') {
        formNodes.push('FOUND_FORM');
      }
      if (ancestor.memoizedProps && ancestor.memoizedProps.onSubmit) {
        formNodes.push('found_onSubmit');
      }
      ancestor = ancestor.return;
    }
    return {
      text: btn.innerText,
      reactKeys: keys.filter(k => k.startsWith('__react')),
      onClick_type: typeof props.onClick,
      onSubmit_type: typeof props.onSubmit,
      type: btn.type,
      formAncestors: formNodes,
    };
  })()
""", 10)
print("React inspect:", info)

# Find form in DOM walking up parents
form_info = ev("""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => /Submit to/.test(b.innerText));
    let el = btn;
    let path = [];
    while (el && path.length < 15) {
      path.push(`${el.tagName}${el.className ? '.' + el.className.split(' ')[0] : ''}`);
      el = el.parentElement;
    }
    return path;
  })()
""", 11)
print("\nDOM ancestor path:", form_info)

# Find ALL forms on page
forms = ev("""
  (() => {
    const forms = Array.from(document.querySelectorAll('form'));
    return forms.map(f => ({id: f.id, action: f.action, method: f.method, n_inputs: f.querySelectorAll('input,textarea').length}));
  })()
""", 12)
print("\nForms on page:", forms)

# Walk react fiber to find onSubmit handler
walk_result = ev("""
  (() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => /Submit to/.test(b.innerText));
    const fiberKey = Object.keys(btn).find(k => k.startsWith('__reactFiber$'));
    let fiber = btn[fiberKey];
    let info = [];
    while (fiber && info.length < 20) {
      const t = fiber.type;
      const name = t ? (t.displayName || t.name || (typeof t === 'string' ? t : '?')) : 'unknown';
      const props = fiber.memoizedProps || {};
      info.push({
        name,
        has_onSubmit: typeof props.onSubmit === 'function',
        has_onClick: typeof props.onClick === 'function',
      });
      fiber = fiber.return;
    }
    return info;
  })()
""", 13)
print("\nReact fiber walk:")
for i, f in enumerate(walk_result or []):
    print(f"  [{i}] {f}")

ws.close()
