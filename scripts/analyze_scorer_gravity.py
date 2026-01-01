
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_scorer_split(player_name_fragment="Draisaitl"):
    print(f"Analyzing Scorer Split for: {player_name_fragment}...")
    
    # 1. Load Data
    try:
        df_grav = pd.read_csv('data/edge_goals/gravity_analysis.csv')
        df_meta = pd.read_csv('data/edge_goals/metadata.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Find Player
    matches = df_grav[df_grav['player_name'].str.contains(player_name_fragment, case=False, na=False)]
    if matches.empty:
        print("Player not found.")
        return
    
    pid = matches['player_id'].iloc[0]
    pname = matches['player_name'].iloc[0]
    print(f"Player: {pname} (ID: {pid})")
    
    # 3. Merge Scorer Info
    # Ensure ID columns match type
    df_grav['game_id'] = df_grav['game_id'].astype(str)
    df_grav['event_id'] = df_grav['event_id'].astype(str)
    df_meta['game_id'] = df_meta['game_id'].astype(str)
    df_meta['event_id'] = df_meta['event_id'].astype(str)
    
    # Create unique event key
    df_grav['key'] = df_grav['game_id'].astype(str) + "_" + df_grav['event_id'].astype(str)
    df_meta['key'] = df_meta['game_id'].astype(str) + "_" + df_meta['event_id'].astype(str)
    
    # Map key -> scorer_id
    scorer_map = pd.Series(df_meta.scorer_id.values, index=df_meta.key).to_dict()
    
    # 4. Filter to Player's Rows
    p_rows = df_grav[df_grav['player_id'] == pid].copy()
    
    # Add scorer column
    p_rows['scorer_id'] = p_rows['key'].map(scorer_map)
    
    # Split
    # Convert everything to float for safe comparison
    p_rows['is_scorer'] = p_rows['scorer_id'].fillna(-1).astype(float) == float(pid)
    
    # Direct Comparison Check
    print(f"Value counts of is_scorer: \n{p_rows['is_scorer'].value_counts()}")
    print(f"Direct check for first 10 rows: { (p_rows['scorer_id'].fillna(-1).astype(float) == float(pid)).tolist()[:10] }")

    scored_by_me = p_rows[p_rows['is_scorer']]
    scored_by_other = p_rows[~p_rows['is_scorer']]
    
    print(f"\n--- Results ---")
    print(f"Total Goals on Ice: {len(p_rows)}")
    print(f"Goals Scored by {pname}: {len(scored_by_me)}")
    print(f"Goals Scored by Teammates: {len(scored_by_other)}")

    # DEBUG TO FILE
    with open('drai_debug.txt', 'w') as f:
        f.write(f"PID: {pid} (type: {type(pid)})\n")
        f.write(f"Sample meta scorer_ids: {df_meta['scorer_id'].head(10).tolist()}\n")
        f.write(f"Sample p_rows scorer_ids: {p_rows['scorer_id'].head(10).tolist()}\n")
        f.write(f"First 10 p_rows keys: {p_rows['key'].head(10).tolist()}\n")
        f.write(f"First 10 meta keys: {df_meta['key'].head(10).tolist()}\n")

    # Check for NaNs in the data
    print(f"NaNs in gravity column: {p_rows['rel_off_puck_mean_dist_ft'].isna().sum()}")
    print(f"Zero weights: {(p_rows['off_puck_frames'] == 0).sum()}")
    
    # DEBUG TYPES
    print("\n--- DEBUG DATA ---")
    print(p_rows[['off_puck_frames', 'rel_off_puck_mean_dist_ft']].dtypes)
    print(p_rows[['off_puck_frames', 'rel_off_puck_mean_dist_ft']].head())
    
    # Ensure numeric
    convert_cols = ['rel_off_puck_mean_dist_ft', 'off_puck_frames']
    for c in convert_cols:
        scored_by_me[c] = pd.to_numeric(scored_by_me[c], errors='coerce')
        scored_by_other[c] = pd.to_numeric(scored_by_other[c], errors='coerce')

    # Weighted Means
    def w_mean(d, col):
        # Filter for valid off-puck frames AND valid gravity values
        valid = d[(d['off_puck_frames'] > 0) & (d[col].notna())]
        if valid.empty: return np.nan
        return np.average(valid[col], weights=valid['off_puck_frames'])
    
    g_me = w_mean(scored_by_me, 'rel_off_puck_mean_dist_ft')
    g_team = w_mean(scored_by_other, 'rel_off_puck_mean_dist_ft')
    
    print(f"\nRelative Off-Puck Gravity (Positive = Open Space):")
    print(f"On Goals He Scored: {g_me:.2f} ft")
    print(f"On Goals by Teammates: {g_team:.2f} ft")
    
    if g_me > 1.0:
        print("\nCONCLUSION: SKILL.")
        print(f"When {pname} scores, he creates SIGNIFICANT positive space ({g_me:.2f} ft).")
        print("He uses this 'Anti-Gravity' to get open for the shot.")
    elif g_me < -1.0:
        print("\nCONCLUSION: BATTLE.")
        print(f"When {pname} scores, he fights through tight checking ({g_me:.2f} ft).")
    else:
        print("\nCONCLUSION: AVERAGE.")

if __name__ == "__main__":
    analyze_scorer_split("Draisaitl")
