"""Check runs via API + try creating one if possible."""
import os, json, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\Admin\earn5usd\crunchdao\datacrunch-2-curly-crayfishwetminh-v2')
os.chdir(r'C:\Users\Admin\earn5usd\crunchdao\datacrunch-2-curly-crayfishwetminh-v2')

token = open('.crunchdao/token').read().strip()
proj = json.loads(open('.crunchdao/project.json').read())
print(f"Token (truncated): {token[:30]}...")
print(f"Project: {proj}")

# Direct HTTP via requests
import requests
BASE = "https://api.hub.crunchdao.com"
headers = {"X-Crunchdao-Auth": token}

# List runs
url = f"{BASE}/v3/competitions/{proj['competitionName']}/projects/{proj['userId']}/{proj['projectName']}/runs"
print(f"\nGET {url}")
r = requests.get(url, headers=headers, params={"submissionNumber": 1}, timeout=10)
print(f"Status: {r.status_code}")
print(f"Body: {r.text[:500]}")

# Try ALL header variants
for header_name in ["Authorization", "X-API-Key", "X-Crunchdao-Token", "X-Token", "Cookie"]:
    h = {header_name: f"Bearer {token}" if header_name == "Authorization" else token}
    if header_name == "Cookie":
        h = {"Cookie": f"crunchdao_token={token}"}
    rr = requests.get(url, headers=h, timeout=5)
    print(f"  {header_name}: {rr.status_code}")
    if rr.status_code == 200:
        print(f"    -> {rr.text[:200]}")
        break
