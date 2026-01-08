
import json

path = "data/arena_adjustments.json"
with open(path, 'r') as f:
    data = json.load(f)
season_2018 = data.get('20182019', {})
print(f"Keys in 20182019: {list(season_2018.keys())[:5]}...")

lightning_2018 = season_2018.get('Lightning', {}).get('x', {})
recorded_85 = lightning_2018.get('85')

print(f"Tampa Bay Lightning (2018-2019) X=85 adjustment: {recorded_85} ft")

# Let's also see the curve around it
for x in range(80, 91):
    print(f"X={x}: {lightning_2018.get(str(x))} ft")
