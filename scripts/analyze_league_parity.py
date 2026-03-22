import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing, analyze, data_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_available_seasons():
    data_dir = Path("data")
    seasons = []
    if data_dir.exists():
        for d in data_dir.iterdir():
            if d.is_dir() and d.name.isdigit() and len(d.name) == 8:
                seasons.append(d.name)
    return sorted(seasons)

def process_season_parity(season):
    logger.info(f"Analyzing parity for {season}...")
    csv_path = analyze.locate_season_csv(season)
    df = pd.read_csv(csv_path)
    
    # Extract unique games and their final scores
    # We can use a simplified version of the logic in evaluate_predictive_power.py
    games = []
    for gid, group in df.groupby('game_id'):
        home_team = group['home_abb'].iloc[0]
        away_team = group['away_abb'].iloc[0]
        home_id = group['home_id'].iloc[0]
        away_id = group['away_id'].iloc[0]
        
        # Regulation outcomes
        home_goals_reg = len(group[(group['event'].str.lower() == 'goal') & (group['team_id'] == home_id) & (group['period'] <= 3)])
        away_goals_reg = len(group[(group['event'].str.lower() == 'goal') & (group['team_id'] == away_id) & (group['period'] <= 3)])
        
        # Ultimate outcomes (including OT/SO)
        # We'll use the max period and total goals as a heuristic if schedule is not available
        home_goals_final = len(group[(group['event'].str.lower() == 'goal') & (group['team_id'] == home_id)])
        away_goals_final = len(group[(group['event'].str.lower() == 'goal') & (group['team_id'] == away_id)])
        is_ot_so = group['period'].max() > 3
        
        # If it's a tie in goals but marked as OT/SO, one team must have "won" the shootout
        # In our data, the shootout winner usually gets an extra goal event or we check the final score
        # For parity, the EXACT score matters less than the margin and the winner.
        
        games.append({
            'game_id': gid,
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals_final,
            'away_goals': away_goals_final,
            'is_ot_so': is_ot_so,
            'margin': abs(home_goals_final - away_goals_final)
        })
    
    sched_df = pd.DataFrame(games)
    
    # Calculate Team Stats
    teams = pd.concat([sched_df['home_team'], sched_df['away_team']]).unique()
    team_stats = []
    for t in teams:
        home_games = sched_df[sched_df['home_team'] == t]
        away_games = sched_df[sched_df['away_team'] == t]
        
        t_games = len(home_games) + len(away_games)
        if t_games == 0: continue
        
        # Points: 2 for win, 1 for OT loss, 0 for Reg loss
        points = 0
        goals_for = 0
        goals_against = 0
        
        for _, row in home_games.iterrows():
            goals_for += row['home_goals']
            goals_against += row['away_goals']
            if row['home_goals'] > row['away_goals']:
                points += 2
            elif row['is_ot_so']:
                points += 1
                
        for _, row in away_games.iterrows():
            goals_for += row['away_goals']
            goals_against += row['home_goals']
            if row['away_goals'] > row['home_goals']:
                points += 2
            elif row['is_ot_so']:
                points += 1
        
        team_stats.append({
            'team': t,
            'points_pct': points / (2 * t_games),
            'gd_per_game': (goals_for - goals_against) / t_games
        })
        
    ts_df = pd.DataFrame(team_stats)
    
    # Parity Metrics
    points_pct_std = ts_df['points_pct'].std()
    gd_std = ts_df['gd_per_game'].std()
    one_goal_pct = (sched_df['margin'] == 1).mean()
    ot_so_pct = sched_df['is_ot_so'].mean()
    avg_goals_per_game = (sched_df['home_goals'].sum() + sched_df['away_goals'].sum()) / len(sched_df)
    
    return {
        'season': season,
        'points_pct_std': points_pct_std,
        'gd_per_game_std': gd_std,
        'one_goal_game_pct': one_goal_pct,
        'ot_so_game_pct': ot_so_pct,
        'avg_goals_per_game': avg_goals_per_game,
        'n_games': len(sched_df)
    }

def main():
    seasons = get_available_seasons()
    # Filter for 20202021 onwards
    seasons = [s for s in seasons if int(s) >= 20202021]
    
    results = []
    for s in seasons:
        try:
            res = process_season_parity(s)
            results.append(res)
        except Exception as e:
            logger.error(f"Error processing {s}: {e}")
            
    res_df = pd.DataFrame(results)
    
    # Save to analysis
    out_dir = Path("analysis")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "parity_metrics.csv"
    res_df.to_csv(out_path, index=False)
    logger.info(f"Parity metrics saved to {out_path}")
    print(res_df)

if __name__ == "__main__":
    main()
