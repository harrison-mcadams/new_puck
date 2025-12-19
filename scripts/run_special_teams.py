import sys
import os
import json
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

# Ensure matplotlib backend
import matplotlib
matplotlib.use('Agg')

from puck import analyze, config

def run():
    season = '20252026'
    print(f"Running Special Teams Analysis for {season}...")
    
    # Load teams
    teams_json_path = os.path.join(config.ANALYSIS_DIR, 'teams.json')
    if os.path.exists(teams_json_path):
        with open(teams_json_path, 'r') as f:
            teams_data = json.load(f)
        teams = sorted([t['abbr'] for t in teams_data if 'abbr' in t])
    else:
        # Fallback
        teams = ['ANA', 'BOS', 'BUF', 'CAR', 'CBJ', 'CGY', 'CHI', 'COL', 'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NJD', 'NSH', 'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'SEA', 'SJS', 'STL', 'TBL', 'TOR', 'UTA', 'VAN', 'VGK', 'WPG', 'WSH']
        
    out_dir = os.path.join(config.ANALYSIS_DIR, 'league', season)
    
    try:
        analyze.generate_special_teams_plot(season, teams, out_dir)
        print("Special Teams plots generated successfully.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run()
