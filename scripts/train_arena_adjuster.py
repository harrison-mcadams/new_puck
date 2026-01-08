
import os
import json
import glob
import numpy as np
import pandas as pd
import argparse
from scipy import interpolate

# Configuration
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "arena_adjustments.json")

def load_season_shots(season_dir):
    """Loads all valid shots from a season directory."""
    shots = []
    
    files = glob.glob(os.path.join(season_dir, "game_*.json"))
    print(f"  Found {len(files)} games in {season_dir}...")
    
    for f_path in files:
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # handle both liveData structure and flat structure
            game_data = data.get('gameData', {})
            home_team = game_data.get('teams', {}).get('home', {}).get('name', 'Unknown')
            # Fallback for old structure
            if home_team == 'Unknown':
                 home_team_obj = data.get('homeTeam', {})
                 home_team = home_team_obj.get('name')
                 if not home_team:
                      home_team = home_team_obj.get('commonName', {}).get('default', 'Unknown')

            # Arena (Venue)
            venue = game_data.get('venue', {}).get('name')
            
            # We strictly need home team name because that's usually the 'Arena' proxy 
            # if venue name changes (e.g. Staples Center -> Crypto.com).
            # Using Home Team as "Arena Key" is often more stable for "Rink Bias".
            # Let's use Home Team Name as the key.
            arena_key = home_team 
            
            if arena_key == 'Unknown':
                 continue 
            
            plays = data.get('plays', [])
            if not plays:
                plays = data.get('liveData', {},).get('plays', {}).get('allPlays', [])
                
            for play in plays:
                event = play.get('typeDescKey')
                if event not in ['shot-on-goal', 'missed-shot', 'goal']:
                    continue
                
                details = play.get('details', {})
                x = details.get('xCoord')
                y = details.get('yCoord')
                
                if x is None or y is None:
                    continue
                
                # Standardize coordinates to one side of the ice?
                # Shuckers/Curro typically adjust 'Distance' and 'X/Y' relative to net.
                # NHL coordinates are -100 to 100. 
                # To detect "recording distance bias", we should look at absolute distance from center (0,0) 
                # or absolute distance from nearest net.
                # Simplest adjustment: Absolute X and Absolute Y distributional matching.
                
                shots.append({
                    'x': abs(x), # Treat all quadrants symmetrically for bias detection (bias is usually "too close/far from net")
                    'y': abs(y), # Same for Y (bias is usually "too center/wide")
                    'arena': arena_key
                })
                
        except Exception as e:
            continue
            
    return pd.DataFrame(shots)

def calculate_adjustments(df):
    """Calculates CDF-based adjustments for each arena compared to league avg."""
    adjustments = {}
    
    # 1. League Distributions (Ground Truth Baseline)
    league_x = np.sort(df['x'].values)
    league_y = np.sort(df['y'].values)
    
    # Create interpolation functions for League Percentiles
    # Given a percentile (0-1), what is the X value?
    # We actually need: Given a Value, what is the Percentile? -> Map to League Value.
    
    # Better approach for Lookup Table:
    # For every integer coordinate 0..100, find its percentile in Arena, 
    # then find the value at that same percentile in League.
    # Adjustment = LeagueValue - ArenaValue
    
    arenas = df['arena'].unique()
    
    for arena in arenas:
        arena_shots = df[df['arena'] == arena]
        if len(arena_shots) < 500: # Skip low sample size
            continue
            
        arena_adj = {'x': {}, 'y': {}}
        
        for coord in ['x', 'y']:
            arena_vals = np.sort(arena_shots[coord].values)
            league_vals = np.sort(df[coord].values) # Or exclude this arena? Standard is usually "League Average" (inclusive)
            
            # We want to map: Arena Value -> League Value
            # Quantile Mapping
            
            # Create interpolator for Arena: Value -> Quantile
            # Note: We handle duplicates by using unique values for interpolation or just using searchsorted
            
            # Genererate a lookup table for integer coordinates
            max_val = 100 if coord == 'x' else 42 # Rink width
            
            for v in range(0, int(max_val) + 1):
                # 1. What percentile is 'v' in this arena?
                # percent = (fraction of shots <= v)
                pct = (np.searchsorted(arena_vals, v, side='right') / len(arena_vals))
                
                # 2. What value is at 'pct' in the League?
                # equivalent to np.percentile(league_vals, pct * 100)
                if pct >= 1.0: target_v = league_vals[-1]
                elif pct <= 0.0: target_v = league_vals[0]
                else: target_v = np.percentile(league_vals, pct * 100)
                
                # 3. Adjustment
                delta = target_v - v
                arena_adj[coord][str(v)] = round(float(delta), 2)
        
        adjustments[arena] = arena_adj
        
    return adjustments

def summarize_findings(adjustments):
    """Prints out the most biased arenas."""
    print("\n--- FINDINGS: ARENA BIAS REPORT ---")
    
    # Calculate 'Average Absolute Adjustment' for X (Distance from Center/Net)
    # Higher X = Closer to Net (since we used abs(x) and nets are at ends... wait.)
    # NHL Coords: 0,0 is center ice. Nets are at -89 and +89.
    # So abs(x) -> 0 is center, 89 is net.
    # If Arena records shots as 80 (closer to center) but League says 85 (closer to net),
    # The bias matches them.
    
    # Let's look at the adjustment at the "Slot" (e.g., 20ft from net => X=69)
    # Key point: X=69 (89-20).
    
    slot_x = "69" 
    
    arena_biases = []
    
    for arena, adj in adjustments.items():
        if 'x' not in adj: continue
        
        # Check adjustment at X=69 (approx 20ft from net)
        # If delta is POSITIVE (+5), it means Arena says 69, League says 74.
        # Adjusted = 74. Arena recorded it "Too Far from Net" (Closer to center).
        # If delta is NEGATIVE (-5), it means Arena says 69, League says 64.
        # Adjusted = 64. Arena recorded it "Too Close to Net".
        
        x_delta = adj['x'].get(slot_x, 0)
        arena_biases.append((arena, x_delta))
        
    print(f"\nBias at ~20ft from Net (X={slot_x}):")
    print("(Negative = Arena records shots too close to net)")
    print("(Positive = Arena records shots too far from net)")
    print("-" * 50)
    
    # Sort by bias
    sorted_biases = sorted(arena_biases, key=lambda x: x[1])
    
    print("TOP 5 'Generous' Arenas (Record shots closer than reality):")
    for a, d in sorted_biases[:5]:
        print(f"  {a.ljust(30)}: {d:+0.2f} ft adjustment needed")
        
    print("\nTOP 5 'Stingy' Arenas (Record shots further than reality):")
    for a, d in sorted_biases[-5:]:
        print(f"  {a.ljust(30)}: {d:+0.2f} ft adjustment needed")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seasons', nargs='+', default=['20232024'], help='Seasons to train on')
    args = parser.parse_args()
    
    master_adjustments = {}
    
    for season in args.seasons:
        print(f"Processing Season {season}...")
        season_dir = os.path.join(DATA_DIR, season)
        
        if not os.path.exists(season_dir):
            print(f"  Skipping {season} (Not found)")
            continue
            
        df = load_season_shots(season_dir)
        if df.empty:
            print("  No shots found.")
            continue
            
        print(f"  Loaded {len(df)} shots.")
        
        # Calculate Adjustments
        adjs = calculate_adjustments(df)
        master_adjustments[season] = adjs
        
        # Summarize just for this season
        summarize_findings(adjs)
        
    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(master_adjustments, f, indent=2)
    print(f"\nSaved adjustments to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
