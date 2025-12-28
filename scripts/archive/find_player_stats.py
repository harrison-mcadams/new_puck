
import json
import sys
import os

def find_player(name_query):
    path = 'analysis/players/20252026/player_summary_5v5.json'
    if not os.path.exists(path):
        print("Summary file not found.")
        return

    with open(path, 'r') as f:
        data = json.load(f)
        
    print(f"Searching {len(data)} players for '{name_query}'...")
    
    found = False
    for pid, stats in data.items():
        pname = stats.get('player_name', 'Unknown')
        if name_query.lower() in pname.lower():
            print(f"\n--- Found: {pname} (ID: {pid}) ---")
            print(json.dumps(stats, indent=2))
            found = True
            
    if not found:
        print("Player not found.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        find_player(sys.argv[1])
    else:
        find_player("Sennecke")
