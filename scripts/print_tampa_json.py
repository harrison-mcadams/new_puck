
import json
path = "data/arena_adjustments.json"
with open(path, 'r') as f:
    data = json.load(f)

tampa = data.get('20182019', {}).get('Lightning', {}).get('x', {})
# Or maybe the key is 'Tampa Bay Lightning'? 
# My script find_top_biases.py used 'Lightning' in the output earlier but maybe it was 'Tampa Bay Lightning'.
# Let's check both
adj = data.get('20182019', {}).get('Lightning', {}).get('x', {})
if not adj:
    adj = data.get('20182019', {}).get('Tampa Bay Lightning', {}).get('x', {})

print("Tampa 2018-2019 X-Adjustments (Abs X -> Delta):")
for x in range(0, 101, 10):
    print(f"X={x}: {adj.get(str(x))}")
print(f"X=85: {adj.get('85')}")
