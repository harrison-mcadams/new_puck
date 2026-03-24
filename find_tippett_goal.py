import pandas as pd
import os

csv_path = r'c:\Users\harri\Desktop\new_puck\data\20252026.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    
    # User specified: Owen Tippett, 2nd period, 2:26
    # period_time: '02:26'
    target_period = 2
    target_time = '02:26'
    target_player = 'Owen Tippett'
    
    match = df[(df['player_name'] == target_player) & 
               (df['period'] == target_period) & 
               (df['period_time'] == target_time)]
    
    if not match.empty:
        print("FOUND MATCHING EVENT:")
        # Show all columns to get feature values
        for i, row in match.iterrows():
            print(row.to_dict())
            print("-" * 20)
    else:
        print("No exact match found for Tippett at 2:26 of P2.")
        # Relax constraints to find it
        nearby = df[(df['player_name'] == target_player) & 
                    (df['period'] == target_period)]
        print(f"Found {len(nearby)} events for Tippett in P2. Searching for goal near 2:26...")
        goals_p2 = nearby[nearby['event'] == 'goal']
        if not goals_p2.empty:
            print("Goals in P2 for Tippett:")
            print(goals_p2[['period_time', 'event', 'game_id']])
        else:
            print("No goals for Tippett in P2.")
            # Search all P2 events around 2:26
            p2_events = df[(df['period'] == target_period) & (df['period_time'] == target_time)]
            print(f"All events at 2:26 of P2 ({len(p2_events)} total):")
            print(p2_events[['player_name', 'event', 'home_abb', 'away_abb']])
else:
    print(f"{csv_path} not found")
