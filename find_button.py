import re
html = open(r"C:\Users\Admin\earn5usd\crunchdao\dom_dump.html", encoding="utf-8").read()
# Find context around "Getting started"
idx = html.find("Getting started")
print(f"Index: {idx}")
if idx > 0:
    print("=== 500 chars before ===")
    print(html[max(0,idx-500):idx])
    print("=== 500 chars after ===")
    print(html[idx:idx+500])
print()
print("=== Search 'rules' ===")
for m in re.finditer(r"rules", html, re.IGNORECASE):
    print(f"  at {m.start()}: ...{html[max(0,m.start()-100):m.end()+100]}...")
    if m.start() > 50000: break
