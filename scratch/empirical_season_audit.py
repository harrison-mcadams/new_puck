import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from puck import config, analyze

def main():
    print("Starting empirical season-by-season audit...")
    data_dir = Path(config.DATA_DIR)
    
    # Discover all season directories/files
    seasons = []
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.isdigit() and len(item.name) == 8:
            seasons.append(item.name)
            
    seasons = sorted(seasons)
    print(f"Discovered {len(seasons)} seasons: {seasons}")
    
    records = []
    
    for s in seasons:
        try:
            csv_path = analyze.locate_season_csv(s)
            if not csv_path:
                print(f"  {s}: CSV not found.")
                continue
                
            print(f"  Processing {s}...")
            # Load only columns we need to save memory
            use_cols = ['event', 'x', 'y', 'distance', 'angle_deg', 'shot_type', 'is_net_empty']
            df = pd.read_csv(csv_path, usecols=lambda x: x in use_cols, low_memory=False)
            
            # Filter for shot events
            shot_events = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
            df = df[df['event'].isin(shot_events)].copy()
            
            total_shots = len(df)
            if total_shots == 0:
                print(f"    No shot events found in {s}.")
                continue
                
            # Exclude empty net shots if column exists
            if 'is_net_empty' in df.columns:
                df = df[~(df['is_net_empty'] == 1)].copy()
                
            goals = (df['event'] == 'goal').sum()
            blocks = (df['event'] == 'blocked-shot').sum()
            shots_on_goal = (df['event'] == 'shot-on-goal').sum()
            misses = (df['event'] == 'missed-shot').sum()
            
            unblocked = total_shots - blocks
            on_net = shots_on_goal + goals
            
            mean_dist = df['distance'].mean() if 'distance' in df.columns else np.nan
            mean_angle = df['angle_deg'].mean() if 'angle_deg' in df.columns else np.nan
            
            # Coordinate counts/NaNs
            nan_x = df['x'].isna().sum() if 'x' in df.columns else total_shots
            nan_y = df['y'].isna().sum() if 'y' in df.columns else total_shots
            
            # Shot Type unknown fraction
            unknown_shot_type = 0
            if 'shot_type' in df.columns:
                # fillna and normalize string
                st_series = df['shot_type'].fillna('Unknown').str.lower()
                unknown_shot_type = st_series.isin(['unknown', '']).sum()
            else:
                unknown_shot_type = total_shots
                
            records.append({
                'season': s,
                'total_attempts': total_shots,
                'goals': goals,
                'blocks': blocks,
                'shots_on_net': on_net,
                'misses': misses,
                'block_rate': blocks / total_shots if total_shots > 0 else np.nan,
                'accuracy_rate': on_net / unblocked if unblocked > 0 else np.nan,
                'shooting_pct': goals / on_net if on_net > 0 else np.nan,
                'goal_attempt_pct': goals / total_shots if total_shots > 0 else np.nan,
                'mean_distance': mean_dist,
                'mean_angle': mean_angle,
                'nan_coords_pct': (nan_x + nan_y) / (2 * total_shots),
                'unknown_shot_type_pct': unknown_shot_type / total_shots
            })
            print(f"    Attempts: {total_shots}, Blocks: {blocks} ({blocks/total_shots:.1%}), Acc: {on_net/unblocked:.1%}, Sh%: {goals/on_net:.1%}")
            
        except Exception as e:
            print(f"  Error processing season {s}: {e}")
            import traceback
            traceback.print_exc()
            
    summary_df = pd.DataFrame(records)
    output_path = project_root / 'analysis' / 'empirical_seasons_summary.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    
    print("\n============================================================")
    print("EMPIRICAL SEASON SUMMARY")
    print("============================================================")
    print(summary_df.to_string(index=False, columns=[
        'season', 'total_attempts', 'block_rate', 'accuracy_rate', 'shooting_pct', 'mean_distance', 'unknown_shot_type_pct'
    ]))
    print(f"\nSaved empirical summary to {output_path}")

if __name__ == "__main__":
    main()
