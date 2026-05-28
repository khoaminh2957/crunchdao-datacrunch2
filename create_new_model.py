"""Click 'New model' in the Model dropdown, capture the POST endpoint that creates it."""
import json, urllib.request, urllib.parse, websocket, time, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tabs = [t for t in tabs if "datacrunch-2/submit" in t.get("url","") and t.get("type") == "page"]
if not tabs:
    # Open new
    req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote('https://hub.crunchdao.com/competitions/datacrunch-2/submit')}", method="PUT")
    urllib.request.urlopen(req); time.sleep(8)
    tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
    tabs = [t for t in tabs if "datacrunch-2/submit" in t.get("url","") and t.get("type") == "page"]

ws = websocket.create_connection(tabs[0]["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, m, p, mid):
    ws.send(json.dumps({"id":mid,"method":m,"params":p or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def ev(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression":expr,"returnByValue":True,"awaitPromise":True}, mid)
    if "exceptionDetails" in r.get("result",{}):
        print("JS_ERR:", json.dumps(r["result"]["exceptionDetails"])[:300])
    return r.get("result",{}).get("result",{}).get("value")

cdp(ws, "Network.enable", {}, 1)

# 1. Open the Model dropdown
r = ev(ws, r"""
  (() => {
    // The model selector is probably a button/combobox showing current model name
    const triggers = Array.from(document.querySelectorAll('button')).filter(b => b.innerText.includes('curly-crayfishwetminh-v2'));
    console.log('triggers:', triggers.length);
    // try the first one
    if (triggers.length > 0) {
      triggers[0].click();
      return {opened: true, count: triggers.length};
    }
    return {error: 'no model dropdown'};
  })()
""", 10)
print("OPEN_DROPDOWN:", r)
time.sleep(1)

# 2. Find and click 'New model'
r = ev(ws, r"""
  (async () => {
    await new Promise(r => setTimeout(r, 500));
    const items = Array.from(document.querySelectorAll('button, [role="menuitem"], [role="option"], a, div'));
    const nm = items.find(el => /^New model/i.test(el.innerText.trim()));
    if (!nm) return {error: 'no new model option', sample: items.slice(0,5).map(e => e.innerText.trim().slice(0,40))};
    nm.click();
    return {clicked_new_model: true, txt: nm.innerText.trim()};
  })()
""", 11)
print("NEW_MODEL:", r)
time.sleep(3)

# 3. Examine dialog
state = ev(ws, r"""
({
  url: location.href,
  inputs: Array.from(document.querySelectorAll('input, textarea')).slice(0,10).map(i => ({type: i.type, name: i.name, placeholder: i.placeholder, value: i.value})),
  buttons: Array.from(document.querySelectorAll('button')).filter(b => b.offsetParent !== null).slice(0,20).map(b => b.innerText.trim()).filter(t => t)
})
""", 12)
print("\nDIALOG STATE:")
print(json.dumps(state, indent=2, ensure_ascii=False)[:2500])

# Also screenshot
ss = cdp(ws, "Page.captureScreenshot", {"format":"png"}, 50)
import base64
img = ss.get("result",{}).get("data","")
if img:
    open("C:/Users/Admin/earn5usd/crunchdao/new_model_dialog.png","wb").write(base64.b64decode(img))
    print("\nScreenshot -> new_model_dialog.png")

ws.close()
