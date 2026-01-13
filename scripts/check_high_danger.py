
import pandas as pd
import numpy as np

def main():
    try:
        df = pd.read_csv('analysis/shot_comparison_2025.csv')
        print(f"Loaded {len(df)} rows.")
        
        # Ensure 'xgs' (My Model) and 'xGoal' (MoneyPuck) exist
        if 'xgs' not in df.columns or 'xGoal' not in df.columns:
            print("Missing xgs or xGoal columns.")
            return

        print("\n--- Max xG Values ---")
        print(f"My Model Max:   {df['xgs'].max():.4f}")
        print(f"MoneyPuck Max:  {df['xGoal'].max():.4f}")

        print("\n--- High Danger Distribution (Count & %) ---")
        thresholds = [0.3, 0.5, 0.7, 0.9]
        for t in thresholds:
            my_count = len(df[df['xgs'] > t])
            mp_count = len(df[df['xGoal'] > t])
            print(f"xG > {t}: My={my_count:<5} ({my_count/len(df):.1%}) | MP={mp_count:<5} ({mp_count/len(df):.1%})")

        print("\n--- Analysis of 'High Danger' Shots (MoneyPuck > 0.5) ---")
        # Filter for shots that MoneyPuck considers high danger
        high_danger_mp = df[df['xGoal'] > 0.5]
        print(f"Count of MP > 0.5: {len(high_danger_mp)}")
        
        if len(high_danger_mp) > 0:
            print(f"My Model Mean on these: {high_danger_mp['xgs'].mean():.4f}")
            print(f"My Model Max on these:  {high_danger_mp['xgs'].max():.4f}")
            print(f"My Model > 0.5 count:   {len(high_danger_mp[high_danger_mp['xgs'] > 0.5])}")
            
            # Check for extreme disconnects
            disconnects = high_danger_mp[high_danger_mp['xgs'] < 0.2]
            print(f"\nSevere Disconnects (MP > 0.5, My < 0.2): {len(disconnects)}")
            if len(disconnects) > 0:
                print("Examples:")
                print(disconnects[['game_id', 'event', 'xgs', 'xGoal', 'distance', 'shotType']].head(5))

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
