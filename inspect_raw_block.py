import requests
import json
import sys

# Use the game ID from the user's example if possible, or another recent one
game_id = "2024020151" 
url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"

print(f"Fetching {url}...")
resp = requests.get(url)
data = resp.json()

blocks = [p for p in data.get('plays', []) if p.get('typeDescKey') == 'blocked-shot']

if not blocks:
    print("No blocks found.")
else:
    print(f"Found {len(blocks)} blocks.")
    b = blocks[0]
    print(json.dumps(b, indent=2))
    
    # Also check who the teams are
    home = data.get('homeTeam', {}).get('id')
    away = data.get('awayTeam', {}).get('id')
    print(f"Home Team: {home}, Away Team: {away}")
