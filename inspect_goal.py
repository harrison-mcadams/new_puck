import requests
import json

game_id = "2024020151"
url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
resp = requests.get(url)
data = resp.json()

goals = [p for p in data.get('plays', []) if p.get('typeDescKey') == 'goal']
if goals:
    g = goals[0]
    print(json.dumps({
        'type': g.get('typeDescKey'),
        'period': g.get('periodDescriptor', {}).get('number'),
        'details': g.get('details'),
        'homeDefending': g.get('homeTeamDefendingSide')
    }, indent=2))
else:
    print("No goals.")
