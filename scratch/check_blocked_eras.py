import requests
import json

game_ids = [2010020001, 2015020001, 2020020001, 2024020001]

for gid in game_ids:
    print(f"\nChecking Game ID: {gid}")
    url = f"https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        plays = [p for p in data.get('plays', []) if p.get('typeDescKey') == 'blocked-shot']
        if not plays:
            print("  No blocked shots found.")
            continue
        
        sample = plays[0]
        details = sample.get('details', {})
        print(f"  Event Code: {sample.get('typeDescKey')}")
        print(f"  Details keys: {list(details.keys())}")
        
        # Check shooter/blocker IDs
        shooting_id = details.get('shootingPlayerId')
        blocking_id = details.get('blockingPlayerId')
        player_id = details.get('playerId')
        owner_team = details.get('eventOwnerTeamId')
        
        print(f"  shootingPlayerId: {shooting_id}")
        print(f"  blockingPlayerId: {blocking_id}")
        print(f"  playerId:         {player_id}")
        print(f"  eventOwnerTeamId: {owner_team}")
        
        # Try to find which team is which
        home_team = data.get('homeTeam', {}).get('id')
        away_team = data.get('awayTeam', {}).get('id')
        print(f"  Home Team: {home_team}, Away Team: {away_team}")
        
    except Exception as e:
        print(f"  Error: {e}")
