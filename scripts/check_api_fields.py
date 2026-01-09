import requests
import json

url = 'https://api.nhle.com/stats/rest/en/skater/bios?isAggregate=false&isGame=false&start=0&limit=5&cayenneExp=seasonId=20232024'
resp = requests.get(url)
data = resp.json()
if 'data' in data and len(data['data']) > 0:
    print("Sample Row Keys:", data['data'][0].keys())
    print("Sample Row:", json.dumps(data['data'][0], indent=2))
else:
    print("No data found or error.")
