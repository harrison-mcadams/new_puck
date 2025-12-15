
import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgs
from puck import impute

def main():
    game_id = 2025020483
    print(f"--- Diagnosing Flyers Game {game_id} ---")
    
    # 1. Load Season Data
    season_file = 'data/20252026.csv'
    if not os.path.exists(season_file):
        print(f"Error: {season_file} not found. Run daily.py first.")
        return
        
    df = pd.read_csv(season_file)
    
    # 2. Filter for Game
    df_game = df[df['game_id'] == game_id].copy()
    if df_game.empty:
        print(f"Error: Game {game_id} not found in data.")
        return
        
    print(f"Found {len(df_game)} events for game {game_id}")
    
    # Filter to shot attempts
    events = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
    df_shots = df_game[df_game['event'].isin(events)].copy()
    print(f"Filtered to {len(df_shots)} shot attempts.")
    
    # 3. Load Model
    model_path = 'analysis/xgs/xg_model_nested.joblib'
    try:
        clf = joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 4. Predict
    # Ensure features exist
    if 'shot_type' not in df_shots.columns:
        df_shots['shot_type'] = 'Unknown'
    df_shots['shot_type'] = df_shots['shot_type'].fillna('Unknown')
    if 'game_state' not in df_shots.columns:
        df_shots['game_state'] = '5v5'
        
    print("Applying blocked shot imputation...")
    df_shots = impute.impute_blocked_shot_origins(df_shots, method='mean_6')
        
    try:
        probs = clf.predict_proba(df_shots)[:, 1]
        df_shots['xg'] = probs
    except Exception as e:
        print(f"Prediction failed: {e}")
        return

    # 5. Save Predictions CSV
    cols = ['event', 'shot_type', 'period', 'periodTime_seconds_elapsed', 'team_id', 'x', 'y', 'distance', 'angle_deg', 'is_net_empty', 'xg']
    # Use secondary_type if shot_type is unknown/generic, often has 'Slap Shot' etc.
    if 'secondary_type' not in df_shots.columns:
        df_shots['secondary_type'] = ''
    else:
        cols.insert(1, 'secondary_type')
        
    out_csv = f'analysis/flyers_game_{game_id}_predictions.csv'
    df_shots[cols].to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")
    
    # 6. Generate Comparison Plot (Cumulative xG)
    # Sort by time
    df_shots = df_shots.sort_values('total_time_elapsed_seconds')
    
    teams = df_shots['team_id'].unique()
    plt.figure(figsize=(10, 6))
    
    for team in teams:
        team_shots = df_shots[df_shots['team_id'] == team].copy()
        team_shots['cumulative_xg'] = team_shots['xg'].cumsum()
        
        # Add start point
        x_vals = [0] + team_shots['total_time_elapsed_seconds'].tolist()
        y_vals = [0] + team_shots['cumulative_xg'].tolist()
        
        plt.plot(x_vals, y_vals, label=f"Team {team} xG", linewidth=2)
        
        # Mark goals
        goals = team_shots[team_shots['event'] == 'goal']
        for _, g in goals.iterrows():
            t_sec = g['total_time_elapsed_seconds']
            # Find closest y val (cumulative xg at that point)
            cum_xg = team_shots.loc[g.name, 'cumulative_xg']
            plt.scatter(t_sec, cum_xg, s=100, marker='o', edgecolors='black', zorder=5, label=f"Team {team} Goal" if f"Team {team} Goal" not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title(f"Cumulative xG: Game {game_id}")
    plt.xlabel("Game Seconds")
    plt.ylabel("Expected Goals")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_png = f'analysis/flyers_game_{game_id}_xg_comparison.png'
    plt.savefig(out_png)
    print(f"Saved comparison plot to {out_png}")

if __name__ == "__main__":
    main()
