
import sys
import os
sys.path.append(os.getcwd())
from puck import nhl_api
import pprint

def check_pbp():
    game_id = 2024020118
    event_id = 751
    
    print(f"Checking PBP for {game_id} Event {event_id}...")
    try:
        feed = nhl_api.get_game_feed(game_id)
        # Find event
        found = False
        for play in feed['plays']:
            if play.get('eventId') == event_id:
                print(f"Found Event {event_id}:")
                pprint.pprint(play)
                found = True
                break
        
        if not found:
            print("Event not found in PBP.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_pbp()
