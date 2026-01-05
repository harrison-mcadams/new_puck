import requests
import sys

def get_block_id(game_id, goal_id):
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    try:
        data = requests.get(url).json()
    except:
        print("Error fetching PBP")
        return

    plays = data.get('plays', [])
    
    # scan for goal
    goal_idx = -1
    for i, p in enumerate(plays):
        if str(p.get('eventId')) == str(goal_id):
            goal_idx = i
            break
            
    if goal_idx == -1:
        print(f"Goal {goal_id} not found in Game {game_id}")
        return

    # Look backwards for Block
    # usually within last 5-10 events?
    # Stop if we hit a stoppage or period start?
    
    print(f"Found Goal {goal_id} at index {goal_idx}. Scanning backwards...")
    
    for i in range(goal_idx - 1, max(-1, goal_idx - 20), -1):
        p = plays[i]
        evt = p.get('typeDescKey', '')
        eid = p.get('eventId')
        print(f"  [{eid}] {evt}")
        
        if evt == 'blocked-shot':
            print(f"FOUND BLOCK: {eid}")
            print(f"Details: {p}")
            return eid

    print("No Block found preceding goal.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python find_block_for_goal.py <game_id> <goal_id>")
    else:
        get_block_id(sys.argv[1], sys.argv[2])
