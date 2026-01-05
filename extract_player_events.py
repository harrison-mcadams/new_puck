import pandas as pd
import os

analysis_file = r"c:\Users\harri\Desktop\new_puck\data\edge_goals\gravity_analysis.csv"
output_file = r"c:\Users\harri\Desktop\new_puck\player_events_deep_dive.csv"

if os.path.exists(analysis_file):
    df = pd.read_csv(analysis_file)
    # Noah Cates (8480220) and Matvei Michkov (8484387)
    target_pids = [8480220, 8484387]
    df_targets = df[df['player_id'].isin(target_pids)]
    
    # Save for investigation
    cols = ['season', 'game_id', 'event_id', 'player_name', 'on_puck_mean_dist_ft', 'off_puck_mean_dist_ft', 'rel_on_puck_mean_dist_ft', 'rel_off_puck_mean_dist_ft', 'on_puck_frames', 'off_puck_frames']
    df_targets[cols].to_csv(output_file, index=False)
    print(f"Extracted {len(df_targets)} events to {output_file}")
    
    # Show summary of a few events
    print("\nSample Cates Events (20252026):")
    print(df_targets[(df_targets['player_id'] == 8480220) & (df_targets['season'] == 20252026)].head(5).to_string())
    
    print("\nSample Michkov Events (20252026):")
    print(df_targets[(df_targets['player_id'] == 8484387) & (df_targets['season'] == 20252026)].head(5).to_string())
else:
    print(f"File not found: {analysis_file}")
