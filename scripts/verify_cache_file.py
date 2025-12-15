
import numpy as np
import json
import os

def main():
    path = "data/cache/20252026/partials/2025020007_5v5.npz"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    with np.load(path, allow_pickle=True) as data:
        keys = list(data.keys())
        print(f"Keys: {len(keys)}")
        
        # Check Team Stats
        # Look for a team_ID_stats key
        team_stats_key = next((k for k in keys if k.startswith('team_') and k.endswith('_stats')), None)
        if team_stats_key:
            print(f"\n--- {team_stats_key} ---")
            s_raw = data[team_stats_key]
            # Handle 0-d array wrapping json
            if s_raw.dtype.kind in {'U', 'S'}:
                stats = json.loads(str(s_raw.item()))
            else:
                stats = json.loads(str(s_raw))
            
            print(json.dumps(stats, indent=2))
            
            # Verify Seconds
            sec = stats.get('team_seconds', 0)
            print(f"Team Seconds: {sec}")
            if sec > 0:
                print("[PASS] Team Seconds present.")
            else:
                print("[FAIL] Team Seconds is 0.")
                
            # Verify xG
            xg = stats.get('team_xgs', 0)
            print(f"Team xG: {xg}")
            if 0 < xg < 10:
                print("[PASS] Team xG is reasonable.")
            else:
                print(f"[FAIL?] Team xG {xg} seems extreme.")
        else:
            print("[FAIL] No team stats found.")
            
        # Check Player Stats
        p_stats_key = next((k for k in keys if k.startswith('p_') and k.endswith('_stats')), None)
        if p_stats_key:
            print(f"\n--- {p_stats_key} ---")
            s_raw = data[p_stats_key]
            if s_raw.dtype.kind in {'U', 'S'}:
                stats = json.loads(str(s_raw.item()))
            else:
                stats = json.loads(str(s_raw))
            
            print(json.dumps(stats, indent=2))
            
            # Verify For vs Against
            # In new logic, player stats should have both if 'team' was correctly identified
            # Currently xgs_map splits by 'team' vs 'other'.
            # Did we verify that player xgs_map call uses 'team' that matches the player's team?
            # If so, 'team_xgs' = Player For, 'other_xgs' = Player Against.
            
            p_xg_for = stats.get('team_xgs', 0)
            p_xg_against = stats.get('other_xgs', 0)
            print(f"Player xG For: {p_xg_for}")
            print(f"Player xG Ag:  {p_xg_against}")
            
            if p_xg_against > 0:
                 print("[PASS] Player has xG Against.")
            else:
                 print("[FAIL?] Player xG Against is 0 (Could be true for short shift, but check TOI)")
                 print(f"Player Seconds: {stats.get('team_seconds')}")

if __name__ == "__main__":
    main()
