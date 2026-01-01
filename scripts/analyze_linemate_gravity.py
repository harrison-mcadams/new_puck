
import pandas as pd
import numpy as np
import os

DATA_DIR = r"c:\Users\harri\Desktop\new_puck\data\edge_goals"
INPUT_FILE = os.path.join(DATA_DIR, "gravity_analysis.csv")

def analyze_wowy(target_player_id, partner_player_id, target_name, partner_name):
    print(f"\n--- WOWY Analysis: {target_name} with/without {partner_name} ---")
    
    if not os.path.exists(INPUT_FILE):
        print("Gravity analysis file not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Ensure IDs are strings for comparison
    df['player_id'] = df['player_id'].astype(str)
    t_id = str(target_player_id)
    p_id = str(partner_player_id)
    
    # 1. Group by event to see who was on ice together
    events = df.groupby(['game_id', 'event_id'])
    
    wowy_data = []
    
    for (gid, eid), group in events:
        pids_on_ice = set(group['player_id'].unique())
        
        if t_id in pids_on_ice:
            # We found a goal where our target was on ice
            target_rows = group[group['player_id'] == t_id]
            if target_rows.empty: continue
            target_row = target_rows.iloc[0]
            
            # Check for partner
            is_partner_present = p_id in pids_on_ice
            
            wowy_data.append({
                'game_id': gid,
                'event_id': eid,
                'rel_off_puck': target_row['rel_off_puck_mean_dist_ft'],
                'off_puck_frames': target_row['off_puck_frames'],
                'partner_present': is_partner_present
            })
            
    df_wowy = pd.DataFrame(wowy_data)
    
    if df_wowy.empty:
        print(f"No samples found for {target_name}.")
        return

    # Weighted Mean helper
    def w_mean(sub_df):
        sub_df = sub_df.dropna(subset=['rel_off_puck'])
        sub_df = sub_df[sub_df['off_puck_frames'] > 0]
        if sub_df.empty: return np.nan
        return np.average(sub_df['rel_off_puck'], weights=sub_df['off_puck_frames'])

    with_p = df_wowy[df_wowy['partner_present'] == True]
    without_p = df_wowy[df_wowy['partner_present'] == False]
    
    mean_with = w_mean(with_p)
    mean_without = w_mean(without_p)
    
    print(f"Goals on ice WITH {partner_name}: {len(with_p)}")
    print(f"Goals on ice WITHOUT {partner_name}: {len(without_p)}")
    print(f"\n{target_name} Off-Puck Gravity:")
    print(f"  WITH {partner_name}:    {mean_with:+.2f} ft")
    print(f"  WITHOUT {partner_name}: {mean_without:+.2f} ft")
    if not np.isnan(mean_with) and not np.isnan(mean_without):
        print(f"  Difference:           {mean_with - mean_without:+.2f} ft")
    else:
        print("  Not enough data for comparison.")

if __name__ == "__main__":
    # Draisaitl: 8477934, McDavid: 8478402
    analyze_wowy(8477934, 8478402, "Leon Draisaitl", "Connor McDavid")
    
    # Also do the reverse
    analyze_wowy(8478402, 8477934, "Connor McDavid", "Leon Draisaitl")
