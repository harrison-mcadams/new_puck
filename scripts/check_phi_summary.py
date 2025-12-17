import json
import os
import sys

def main():
    path = "analysis/xleague/20252026/5v5/20252026_team_summary.json"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Structure: list of team dicts or dict of team dicts?
        # Usually list.
        phi = None
        if isinstance(data, list):
            for t in data:
                if t.get('team') == 'PHI':
                    phi = t
                    break
        elif isinstance(data, dict):
             # Maybe keyed by team or has 'teams' key?
             if 'PHI' in data:
                 phi = data['PHI']
             elif 'teams' in data:
                 # Search inside
                 pass
                 
        if phi:
            print("PHI Summary Found:")
            print(json.dumps(phi, indent=2))
            
            goals = phi.get('team_goals', 0)
            xg = phi.get('team_xgs', 0)
            print("-" * 20)
            print(f"Goals: {goals}")
            print(f"xG:    {xg}")
            if goals > 0:
                print(f"Ratio: {xg/goals:.3f}")
        else:
            print("PHI not found in summary.")
            
    except Exception as e:
        print(f"Error reading JSON: {e}")

if __name__ == "__main__":
    main()
