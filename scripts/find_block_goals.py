
import sys
import os
sys.path.append(os.getcwd())
from puck import nhl_api
import pprint

def find_block_goals():
    game_id = 2024020118
    print(f"Scanning Game {game_id} for Goals preceded by Blocks...")
    
    try:
        feed = nhl_api.get_game_feed(game_id)
        plays = feed['plays']
        
        goals = [p for p in plays if p['typeDescKey'] == 'goal']
        
        for goal in goals:
            goal_idx = plays.index(goal)
            goal_id = goal['eventId']
            goal_time = goal['timeInPeriod']
            
            print(f"Checking Goal {goal_id} at {goal_time}...")
            
            # Look back 5 events
            start_idx = max(0, goal_idx - 10)
            for i in range(start_idx, goal_idx):
                e = plays[i]
                if e['typeDescKey'] == 'blocked-shot':
                    print(f"  [CANDIDATE] Found Blocked Shot (Event {e['eventId']}) at {e['timeInPeriod']} before Goal {goal_id}!")
                    print(f"  Sequence: Block {e['eventId']} -> ... -> Goal {goal_id}")
                    return # Found one, stop
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_block_goals()
