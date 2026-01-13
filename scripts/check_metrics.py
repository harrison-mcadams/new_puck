
import pandas as pd
import numpy as np

def main():
    try:
        df = pd.read_csv('analysis/shot_comparison_2025.csv')
        # print(f"Loaded {len(df)} rows.")
        
        # Ensure we have necessary columns
        if 'xgs' not in df.columns or 'xGoal' not in df.columns or 'event' not in df.columns:
            print("Missing required columns (xgs, xGoal, event).")
            return

        # Create 'is_goal' column
        df['is_goal'] = (df['event'] == 'goal').astype(int)
        
        # GLobal Totals
        total_xg_my = df['xgs'].sum()
        total_xg_mp = df['xGoal'].sum()
        total_goals = df['is_goal'].sum()
        
        print("\n--- Global Totals (Matched Data) ---")
        print(f"My Model xG:   {total_xg_my:.2f}")
        print(f"MoneyPuck xG:  {total_xg_mp:.2f}")
        print(f"Actual Goals:  {total_goals}")
        print(f"Diff My-Act:   {total_xg_my - total_goals:.2f} ({((total_xg_my - total_goals)/total_goals)*100:.1f}%)")
        print(f"Diff MP-Act:   {total_xg_mp - total_goals:.2f} ({((total_xg_mp - total_goals)/total_goals)*100:.1f}%)")
        
        # Team Level Comparison
        # Group by 'team_id'
        if 'team_id' in df.columns:
            team_stats = df.groupby('team_id').agg({
                'xgs': 'sum',
                'xGoal': 'sum',
                'is_goal': 'sum',
                'game_id': 'nunique'
            }).reset_index()
            
            team_stats['my_diff'] = team_stats['xgs'] - team_stats['is_goal']
            team_stats['mp_diff'] = team_stats['xGoal'] - team_stats['is_goal']
            
            print("\n--- Team Level Comparison (Top 10 by Abs Diff My Model) ---")
            team_stats['abs_my_diff'] = team_stats['my_diff'].abs()
            top_diff = team_stats.sort_values('abs_my_diff', ascending=False).head(10)
            
            print(f"{'ID':<4} {'Actual':<8} {'My xG':<8} {'MP xG':<8} {'My Diff':<8} {'MP Diff':<8}")
            for _, row in top_diff.iterrows():
                print(f"{row['team_id']:<4} {row['is_goal']:<8.0f} {row['xgs']:<8.2f} {row['xGoal']:<8.2f} {row['my_diff']:<8.2f} {row['mp_diff']:<8.2f}")
                
            print("\n--- League Wide Summary ---")
            print(f"Mean Abs Error (Team Totals) - My: {team_stats['my_diff'].abs().mean():.2f}")
            print(f"Mean Abs Error (Team Totals) - MP: {team_stats['mp_diff'].abs().mean():.2f}")
            
        else:
            print("\n'team_id' column not found, skipping team comparison.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
