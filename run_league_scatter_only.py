import pandas as pd
import os
from run_player_analysis import generate_league_scatter

def run_scatter_only():
    season = '20252026'
    out_dir_base = 'static/players'
    league_out_dir = os.path.join(out_dir_base, f'{season}/league')
    csv_path = os.path.join(league_out_dir, 'league_player_stats.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: Stats file not found at {csv_path}")
        print("Please run run_player_analysis.py first to generate stats.")
        return

    print(f"Loading stats from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("Regenerating league scatter plot...")
    generate_league_scatter(df, league_out_dir, season, min_games=5)
    print("Done.")

if __name__ == "__main__":
    run_scatter_only()
