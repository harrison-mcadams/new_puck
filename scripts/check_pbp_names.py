
import sys
import os
sys.path.append(os.getcwd())
from puck import nhl_api
import pprint

def check_block_317():
    game_id = 2024020118
    block_id = 317
    
    feed = nhl_api.get_game_feed(game_id)
    
    # Map Rosters
    player_map = {}
    for p in feed.get('rosterSpots', []):
        pid = p.get('playerId')
        fname = p.get('firstName', {}).get('default')
        lname = p.get('lastName', {}).get('default')
        if pid: player_map[pid] = f"{fname} {lname}"

    block_play = next((p for p in feed.get('plays', []) if str(p.get('eventId')) == str(block_id)), None)
    
    if block_play:
        print(f"Block Event {block_id}:")
        bd = block_play.get('details', {})
        blk_id = bd.get('blockingPlayerId')
        sht_id = bd.get('shootingPlayerId')
        print(f"  Blocker: {player_map.get(blk_id, blk_id)} (ID: {blk_id})")
        print(f"  Shooter: {player_map.get(sht_id, sht_id)} (ID: {sht_id})")
    # Check Goal 319
    goal_play = next((p for p in feed.get('plays', []) if str(p.get('eventId')) == '319'), None)
    if goal_play:
        print("\nGoal Event 319:")
        gd = goal_play.get('details', {})
        scorer_id = gd.get('scoringPlayerId')
        time = goal_play.get('timeInPeriod')
        print(f"  Time: {time}")
        print(f"  Scorer: {player_map.get(scorer_id, scorer_id)} (ID: {scorer_id})")
        
if __name__ == "__main__":
    check_block_317()
