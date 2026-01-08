
import json

path = "data/arena_adjustments.json"
with open(path, 'r') as f:
    data = json.load(f)

lightning_2018 = data.get('20182019', {}).get('Lightning', {}).get('x', {})
# Sort keys naturally
sorted_keys = sorted(lightning_2018.keys(), key=lambda x: int(x))

print("Tampa Bay Lightning (2018-2019) X-Adjustments:")
for k in sorted_keys:
    print(f"Recorded X={k}ft: Adjustment = {lightning_2018[k]} ft")
