"""Hard click reveal + longer wait + dump full text."""
import json, urllib.request, urllib.parse, websocket, time, sys, re
sys.stdout.reconfigure(encoding='utf-8')
WS_KW = dict(origin="http://localhost", suppress_origin=False)

url = "https://hub.crunchdao.com/competitions/datacrunch-2/submit/cli?projectName=historic-skunk"

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
for t in tabs:
    if "submit/cli" in t.get("url","") and "historic-skunk" in t.get("url",""):
        try: urllib.request.urlopen(f"http://localhost:9222/json/close/{t['id']}")
        except: pass
time.sleep(1)
req = urllib.request.Request(f"http://localhost:9222/json/new?{urllib.parse.quote(url)}", method="PUT")
urllib.request.urlopen(req); time.sleep(12)

tabs = json.loads(urllib.request.urlopen("http://localhost:9222/json").read())
tab = [t for t in tabs if "submit/cli" in t.get("url","") and "historic-skunk" in t.get("url","") and t.get("type") == "page"][0]
ws = websocket.create_connection(tab["webSocketDebuggerUrl"], **WS_KW)

def cdp(ws, m, p, mid):
    ws.send(json.dumps({"id":mid,"method":m,"params":p or {}}))
    while True:
        r = json.loads(ws.recv())
        if r.get("id") == mid: return r

def ev(ws, expr, mid):
    r = cdp(ws, "Runtime.evaluate", {"expression":expr,"returnByValue":True,"awaitPromise":True}, mid)
    return r.get("result",{}).get("result",{}).get("value")

# Click ALL Reveal buttons (CLI tab + Setup commands section)
r = ev(ws, """
  (() => {
    const all = Array.from(document.querySelectorAll('button'));
    const reveal = all.filter(b => /Reveal/i.test(b.innerText));
    // Also click 'Submit via CLI' tab if exists
    const cli = all.find(b => /Submit via CLI/i.test(b.innerText));
    if (cli) cli.click();
    reveal.forEach(x=>x.click());
    return {reveal_clicked: reveal.length, cli_clicked: !!cli};
  })()
""", 5)
print("CLICKED:", r)
time.sleep(4)
r2 = ev(ws, """
  (() => {
    const reveal = Array.from(document.querySelectorAll('button')).filter(b => /Reveal/i.test(b.innerText));
    reveal.forEach(x=>x.click());
    return {reveal_clicked2: reveal.length};
  })()
""", 6)
print("RECLICK:", r2)
time.sleep(3)

txt = ev(ws, "document.body.innerText", 10)
m = re.search(r"crunch setup\s+datacrunch-2\s+historic-skunk\s+--token\s+(\S+)", txt or "")
if m:
    print(f"TOKEN: {m.group(1)}")
else:
    idx = (txt or "").find("crunch setup")
    if idx > 0:
        print("SETUP TEXT:")
        print(txt[idx:idx+600])
    else:
        # Search for any token-like pattern in text
        toks = re.findall(r"[A-Za-z0-9]{60,}", txt or "")
        print(f"No 'crunch setup'. Long-token candidates ({len(toks)}):")
        for t in toks[:5]:
            print(" ", t)
        print("\nFull text:")
        print((txt or ""))
ws.close()
