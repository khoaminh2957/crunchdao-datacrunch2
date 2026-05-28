import json, urllib.request, urllib.parse, websocket, time, base64, sys
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tabs = [t for t in tabs if "submit/cli" in t.get("url","") and "bewildered-chickadee" in t.get("url","") and t.get("type") == "page"]
ws = websocket.create_connection(tabs[0]["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, m, p, mid):
    ws.send(json.dumps({"id":mid,"method":m,"params":p or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def ev(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression":expr,"returnByValue":True,"awaitPromise":True}, mid)
    return r.get("result",{}).get("result",{}).get("value")

state = ev(ws, r"""
({
  url: location.href,
  textLen: document.body.innerText.length,
  text: document.body.innerText.slice(0, 5000),
  hasReveal: !!Array.from(document.querySelectorAll('button')).find(b => /Reveal/i.test(b.innerText)),
  ariaButtons: Array.from(document.querySelectorAll('button')).filter(b => b.offsetParent).map(b => b.innerText.trim()).slice(0, 30)
})
""", 5)
print("URL:", state['url'])
print("TEXTLEN:", state['textLen'])
print("TEXT:\n", state['text'][:3000])
print("\nBUTTONS:", state['ariaButtons'])

ss = cdp(ws, "Page.captureScreenshot", {"format":"png","captureBeyondViewport":True}, 50)
img = ss.get("result",{}).get("data","")
if img:
    open("C:/Users/Admin/earn5usd/crunchdao/cli_page.png","wb").write(base64.b64decode(img))
    print("\nScreenshot -> cli_page.png")
ws.close()
