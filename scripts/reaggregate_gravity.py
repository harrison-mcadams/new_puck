import pandas as pd
import numpy as np
import os

INPUT_FILE = r"c:\Users\harri\Desktop\new_puck\data\edge_goals\gravity_analysis.csv"
OUTPUT_FILE = r"c:\Users\harri\Desktop\new_puck\analysis\gravity\player_gravity_season.csv" # Overwrite the dashboard input

def weighted_mean(df, val_col, weight_col):
    df_sub = df.dropna(subset=[val_col, weight_col])
    if df_sub.empty: return np.nan
    if df_sub[weight_col].sum() == 0: return np.nan
    return np.average(df_sub[val_col], weights=df_sub[weight_col])

def reaggregate():
    print("Loading analysis data...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Could not load data: {e}")
        return

    print(f"Loaded {len(df)} events. Aggregating by frame weights...")

    # Ensure numeric
    cols = ['on_puck_mean_dist_ft', 'off_puck_mean_dist_ft', 'rel_on_puck_mean_dist_ft', 'rel_off_puck_mean_dist_ft', 'on_puck_frames', 'off_puck_frames']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce')

    # Calculate Event-Level Team Averages for adjustment (Position-Specific)
    print("Calculating positionally-adjusted teammate adjustments...")
    # Tag as Defense or Forward
    df['is_defense'] = df['position'].apply(lambda x: 1 if x == 'D' else 0)
    
    # Group by Event AND Position Group
    event_pos_means = df.groupby(['game_id', 'event_id', 'is_defense'])['rel_off_puck_mean_dist_ft'].transform('mean')
    df['group_rel_off_puck_mean'] = event_pos_means
    df['rel_to_teammates_off_puck'] = df['rel_off_puck_mean_dist_ft'] - df['group_rel_off_puck_mean']

    # Group by Player/Season
    results = []
    
    for (season, pid), group in df.groupby(['season', 'player_id']):
        # Basic info
        name = group['player_name'].iloc[0]
        team = group['team_abbr'].iloc[0]
        pos = group['position'].iloc[0]
        goals_count = len(group)

        # Weighted Averages
        on_mean = weighted_mean(group, 'on_puck_mean_dist_ft', 'on_puck_frames')
        on_nearest = weighted_mean(group, 'on_puck_nearest_dist_ft', 'on_puck_frames')
        rel_on = weighted_mean(group, 'rel_on_puck_mean_dist_ft', 'on_puck_frames')
        
        off_mean = weighted_mean(group, 'off_puck_mean_dist_ft', 'off_puck_frames')
        off_nearest = weighted_mean(group, 'off_puck_nearest_dist_ft', 'off_puck_frames')
        rel_off = weighted_mean(group, 'rel_off_puck_mean_dist_ft', 'off_puck_frames')

        # Teammate Adjusted
        rel_to_tm = weighted_mean(group, 'rel_to_teammates_off_puck', 'off_puck_frames')

        results.append({
            'season': season,
            'player_id': pid,
            'player_name': name,
            'team_abbr': team,
            'position': pos,
            'on_puck_mean_dist_ft': on_mean,
            'on_puck_nearest_dist_ft': on_nearest,
            'off_puck_mean_dist_ft': off_mean,
            'off_puck_nearest_dist_ft': off_nearest,
            'rel_on_puck_mean_dist_ft': rel_on,
            'rel_off_puck_mean_dist_ft': rel_off,
            'rel_to_teammates_off_puck': rel_to_tm,
            'goals_on_ice_count': goals_count
        })

    df_agg = pd.DataFrame(results)
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_agg.to_csv(OUTPUT_FILE, index=False)
    print(f"Aggregated {len(df_agg)} players. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    reaggregate()
