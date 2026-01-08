import json
import time
import os
import sys
from typing import Dict, Any, List, Optional

# Add project root to path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck.nhl_api import SESSION, get_game_feed, _throttle

def get_game_boxscore(game_id: int) -> Dict[str, Any]:
    """Fetches the boxscore to get the game's full roster."""
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
    _throttle()
    resp = SESSION.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

def get_player_edge_shots(player_id: int, season: str, game_type: int = 2) -> List[Dict[str, Any]]:
    """Fetches every shot recorded by Edge for a specific player in a season."""
    # Pattern: /v1/edge/skater-shot-speed-detail/{player_id}/{season}/{game_type}
    url = f"https://api-web.nhle.com/v1/edge/skater-shot-speed-detail/{player_id}/{season}/{game_type}"
    _throttle()
    try:
        resp = SESSION.get(url, timeout=10)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
        # Edge response nests individual shots in 'hardestShots'
        return data.get('hardestShots', []) or data.get('data', [])
    except Exception as e:
        print(f"  !! Error fetching shots for {player_id}: {e}")
        return []

def get_team_zone_time(team_id: int, season: str) -> Dict[str, Any]:
    """Fetches aggregate zone time metrics for a team."""
    url = f"https://api-web.nhle.com/v1/edge/team-zone-time/{team_id}/{season}"
    _throttle()
    try:
        resp = SESSION.get(url, timeout=10)
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}

def time_to_seconds(time_str: str) -> int:
    """Converts MM:SS to total seconds."""
    if not isinstance(time_str, str) or ":" not in time_str:
        return 0
    try:
        m, s = map(int, time_str.split(':'))
        return m * 60 + s
    except:
        return 0

def enrich_game(game_id: int):
    print(f"--- Starting Enrichment for Game {game_id} ---")
    
    # 1. Fetch PBP
    pbp = get_game_feed(game_id)
    if not pbp:
        print("Failed to fetch PBP feed.")
        return
    
    season = str(pbp.get('season', '20232024'))
    game_type = int(pbp.get('gameType', 2))
    # Handle both modern and legacy keys
    plays = pbp.get('playByPlay') or pbp.get('plays') or []
    print(f"Fetched {len(plays)} PBP events.")
    
    # 2. Identify roster
    print("Gathering roster from PBP...")
    # Use rosterSpots from PBP if available (most reliable in new API)
    roster_spots = pbp.get('rosterSpots')
    
    player_ids = []
    if roster_spots:
        for spot in roster_spots:
            pid = spot.get('playerId')
            if pid:
                player_ids.append(pid)
    else:
        # Fallback to boxscore if PBP roster is missing
        print("PBP roster missing, falling back to boxscore...")
        try:
            boxscore = get_game_boxscore(game_id)
            player_by_team = boxscore.get('playerByTeam', []) or []
            for team_data in player_by_team:
                for cat in ['forwards', 'defense', 'goalies']:
                    for player in team_data.get(cat, []):
                        player_ids.append(player.get('playerId'))
        except Exception as e:
            print(f"Boxscore fallback failed: {e}")
    
    print(f"Found {len(player_ids)} unique players. Gathering Edge data...")
    
    player_edge_shots = {}
    for pid in player_ids:
        print(f"  -> Pulling Edge shots for player {pid}...")
        player_edge_shots[pid] = get_player_edge_shots(pid, season, game_type)
        time.sleep(0.05) # Gentle throttling
        
    # 3. Merge Logic
    print("Merging Edge shots with PBP events...")
    matched_count = 0
    enriched_plays = []
    
    for play in plays:
        play_type = play.get('typeDescKey')
        # We only care about events with shot speed potential
        if play_type in ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']:
            details = play.get('details', {})
            # Handle different detail keys
            player_id = details.get('shootingPlayerId') or details.get('playerId')
            
            if player_id and player_id in player_edge_shots:
                pbp_time = play.get('timeInPeriod')
                pbp_period = play.get('periodDescriptor', {}).get('number')
                pbp_secs = time_to_seconds(pbp_time)
                
                # Scan player's season edge shots for a match in this game/period/time
                best_match = None
                min_diff = 10 # 10 second window
                
                for eshot in player_edge_shots[player_id]:
                    # Match game and period
                    if eshot.get('gameId') == game_id and eshot.get('period') == pbp_period:
                        e_time = eshot.get('timeInPeriod')
                        e_secs = time_to_seconds(e_time)
                        
                        diff = abs(e_secs - pbp_secs)
                        if diff < min_diff:
                            best_match = eshot
                            min_diff = diff
                
                if best_match:
                    play['edge_shot_speed_mph'] = best_match.get('shotSpeed')
                    play['edge_shot_distance_ft'] = best_match.get('shotDistance')
                    matched_count += 1
        
        enriched_plays.append(play)
    
    print(f"Successfully matched {matched_count} shots with Edge speed data.")
    
    # 4. Save results
    out_file = f"scripts/enriched_pbp_{game_id}.json"
    with open(out_file, 'w') as f:
        json.dump(enriched_plays, f, indent=2)
    
    print(f"Enriched PBP saved to {out_file}")

if __name__ == "__main__":
    # Test with a specific game ID
    target_game = 2023020082 # Rangers @ Flyers
    enrich_game(target_game)
