
import json
import logging
from pathlib import Path
import requests

def inspect():
    # known past game
    game_id = "2025020001"
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    print(f"Fetching {url}...")
    
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Failed to fetch: {e}")
        return

    # Helper to get team info
    home = data.get('homeTeam', {})
    away = data.get('awayTeam', {})
    print(f"Home: {home.get('id')} ({home.get('abbrev')})")
    print(f"Away: {away.get('id')} ({away.get('abbrev')})")

    # Inspect plays
    plays = data.get('plays', []) or data.get('playByPlay', {}).get('plays', [])
    print(f"Plays length: {len(plays)}")
    
    blocked_shots = [p for p in plays if (p.get('typeDescKey') == 'blocked-shot' or (p.get('type') or {}).get('description') == 'Blocked Shot')]
    
    print(f"Found {len(blocked_shots)} blocked shots.\n")

    for i, p in enumerate(blocked_shots[:5]): # Inspect first 5
        print(f"--- Blocked Shot {i+1} ---")
        print(f"Event Owner Team ID: {p.get('details', {}).get('eventOwnerTeamId')}")
        print(f"Team Dict: {p.get('team')}") # 'team' key often missing in plays, usually in details? No, parse.py uses p.get('team') too.
        # But wait, parse.py says: 
        # team_id = details.get('eventOwnerTeamId')
        # team_obj = p.get('team')
        
        print(f"Details: {p.get('details')}")
        
        # Check coordinates
        print(f"Coordinates: {p.get('coordinates')}")
        print(f"Zone: {p.get('details', {}).get('zoneCode')}")

        owner_id = p.get('details', {}).get('eventOwnerTeamId')
        blocker_id = p.get('details', {}).get('blockingPlayerId')
        shooter_id = p.get('details', {}).get('shootingPlayerId')
        
        print(f"  -> Owner Team ID: {owner_id}")
        print(f"  -> Blocker Player ID: {blocker_id}")
        print(f"  -> Shooter Player ID: {shooter_id}")
        
        # We need to look up these players to know their team.
        # The roster is in data['rosterSpots'].
        roster = data.get('rosterSpots', [])
        
        def find_player_team(pid):
            for r in roster:
                if r.get('playerId') == pid:
                    return r.get('teamId')
            return None
            
        blocker_team = find_player_team(blocker_id)
        shooter_team = find_player_team(shooter_id)
        
        print(f"  -> Blocker Team ID: {blocker_team}")
        print(f"  -> Shooter Team ID: {shooter_team}")
        
        if owner_id == shooter_team:
            print("  ==> CONCLUSION: Event Owner is SHOOTER")
        elif owner_id == blocker_team:
            print("  ==> CONCLUSION: Event Owner is BLOCKER")
        else:
            print("  ==> CONCLUSION: Unknown Owner Attribution")

if __name__ == "__main__":
    inspect()
