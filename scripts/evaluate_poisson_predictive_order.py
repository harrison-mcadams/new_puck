"""scripts/evaluate_poisson_predictive_order.py

Unified evaluation routine to compare expected goals (xG) vs. actual goals
for predicting game winners in chronological order within each season using a Poisson distribution.
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from scipy.stats import poisson
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from puck import config, analyze, data_pipeline
from scripts.evaluate_predictive_power import DataUtils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Style settings for premium look
try:
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
except:
    pass
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Outfit', 'DejaVu Sans', 'Arial']

def calculate_poisson_win_prob(h_exp, a_exp):
    """Calculates the Home win probability using Poisson distribution, split 50/50 for ties."""
    h_exp = max(0.01, h_exp)
    a_exp = max(0.01, a_exp)
    
    max_goals = 15
    h_dist = poisson.pmf(np.arange(max_goals), h_exp)
    a_dist = poisson.pmf(np.arange(max_goals), a_exp)
    
    prob_matrix = np.outer(h_dist, a_dist)
    
    home_wins = np.tril(prob_matrix, -1).sum()
    away_wins = np.triu(prob_matrix, 1).sum()
    ties = np.trace(prob_matrix)
    
    total = home_wins + away_wins + ties
    if total > 0:
        home_wins /= total
        ties /= total
        
    return home_wins + 0.5 * ties

def run_season_predictions(season, model, apply_arena_adjustments=True):
    """Runs chronological Poisson predictions for a single season."""
    logger.info(f"Processing season {season}...")
    
    # Load and preprocess season data
    df = DataUtils.load_season_data(season, apply_arena_adjustments=apply_arena_adjustments)
    
    # Ensure numerical season is fed correctly
    df['season'] = int(season)
    
    # Score all shot/goal events in one batch
    logger.info("  Batch-scoring all events with final model...")
    df['eval_xg'] = model.predict_proba(df)[:, 1]
    
    # Extract chronological list of games
    logger.info("  Reconstructing game schedules and outcomes...")
    sched_df = DataUtils.process_schedule(df)
    
    # Map games to team-level totals (expected goals and actual goals scored/conceded per game)
    game_stats = {}
    for gid, group in df.groupby('game_id'):
        home_id = group['home_id'].iloc[0]
        away_id = group['away_id'].iloc[0]
        
        home_xg = group[group['team_id'] == home_id]['eval_xg'].sum()
        away_xg = group[group['team_id'] == away_id]['eval_xg'].sum()
        
        game_stats[gid] = {
            'home_xg': home_xg,
            'away_xg': away_xg
        }
        
    # Sort games chronologically (by game_id)
    sched_df = sched_df.sort_values('game_id').reset_index(drop=True)
    
    # Calculate season-wide league average rates for scaling matchups
    # League Average Goals = total goals / (2 * num_games)
    total_goals = sched_df['home_goals_final'].sum() + sched_df['away_goals_final'].sum()
    league_avg_goals = total_goals / (2 * len(sched_df)) if len(sched_df) > 0 else 3.0
    
    total_xg = sum(v['home_xg'] + v['away_xg'] for v in game_stats.values())
    league_avg_xg = total_xg / (2 * len(sched_df)) if len(sched_df) > 0 else 3.0
    
    logger.info(f"  League Averages - Goals: {league_avg_goals:.3f} | xG: {league_avg_xg:.3f}")
    
    # Track team-level history: dict of lists of historical game outcomes
    teams = pd.concat([sched_df['home_team'], sched_df['away_team']]).unique()
    team_hist = {t: {'actual_for': [], 'actual_against': [], 'xg_for': [], 'xg_against': []} for t in teams}
    
    predictions = []
    
    for _, row in sched_df.iterrows():
        gid = row['game_id']
        h, a = row['home_team'], row['away_team']
        
        h_goals = row['home_goals_final']
        a_goals = row['away_goals_final']
        h_xg = game_stats[gid]['home_xg']
        a_xg = game_stats[gid]['away_xg']
        
        # Chronological games of info available so far
        m_H = len(team_hist[h]['actual_for'])
        m_A = len(team_hist[a]['actual_for'])
        
        # Decide N: average training games of information available
        N = int(np.round((m_H + m_A) / 2))
        
        # --- xG Model Prediction ---
        if m_H == 0 or m_A == 0:
            xg_prob = 0.5
        else:
            h_off_xg = max(0.01, np.mean(team_hist[h]['xg_for']))
            h_def_xg = max(0.01, np.mean(team_hist[h]['xg_against']))
            a_off_xg = max(0.01, np.mean(team_hist[a]['xg_for']))
            a_def_xg = max(0.01, np.mean(team_hist[a]['xg_against']))
            
            lambda_h_xg = (h_off_xg * a_def_xg) / league_avg_xg
            lambda_a_xg = (a_off_xg * h_def_xg) / league_avg_xg
            xg_prob = calculate_poisson_win_prob(lambda_h_xg, lambda_a_xg)
            
        # --- Actual Goals Prediction ---
        if m_H == 0 or m_A == 0:
            goals_prob = 0.5
        else:
            h_off_g = max(0.01, np.mean(team_hist[h]['actual_for']))
            h_def_g = max(0.01, np.mean(team_hist[h]['actual_against']))
            a_off_g = max(0.01, np.mean(team_hist[a]['actual_for']))
            a_def_g = max(0.01, np.mean(team_hist[a]['actual_against']))
            
            lambda_h_g = (h_off_g * a_def_g) / league_avg_goals
            lambda_a_g = (a_off_g * h_def_g) / league_avg_goals
            goals_prob = calculate_poisson_win_prob(lambda_h_g, lambda_a_g)
            
        # Actual Winner
        actual_winner = 1.0 if h_goals > a_goals else 0.0
        
        # Accuracy Evaluation
        # xG
        if xg_prob > 0.5:
            xg_correct = 1.0 if actual_winner == 1.0 else 0.0
        elif xg_prob < 0.5:
            xg_correct = 1.0 if actual_winner == 0.0 else 0.0
        else:
            xg_correct = 0.5 # coinflip expectation
            
        # Goals
        if goals_prob > 0.5:
            goals_correct = 1.0 if actual_winner == 1.0 else 0.0
        elif goals_prob < 0.5:
            goals_correct = 1.0 if actual_winner == 0.0 else 0.0
        else:
            goals_correct = 0.5
            
        predictions.append({
            'season': season,
            'game_id': gid,
            'N': N,
            'xg_correct': xg_correct,
            'goals_correct': goals_correct
        })
        
        # Update team histories with results of this game
        team_hist[h]['actual_for'].append(h_goals)
        team_hist[h]['actual_against'].append(a_goals)
        team_hist[h]['xg_for'].append(h_xg)
        team_hist[h]['xg_against'].append(a_xg)
        
        team_hist[a]['actual_for'].append(a_goals)
        team_hist[a]['actual_against'].append(h_goals)
        team_hist[a]['xg_for'].append(a_xg)
        team_hist[a]['xg_against'].append(h_xg)
        
    return pd.DataFrame(predictions)

def main():
    logger.info("==========================================================")
    logger.info("CHRONOLOGICAL POISSON WINNER PREDICTION EVALUATION ROUTINE")
    logger.info("==========================================================")
    
    # 1. Discover all seasons and load final model
    all_seasons = DataUtils.get_available_seasons()
    modern_seasons = sorted([s for s in all_seasons if int(s) >= 20202021])
    
    model_path = os.path.join('analysis', 'xgs', 'xg_model_xgboost_tensor_final.joblib')
    if not os.path.exists(model_path):
        logger.error(f"Final model not found at {model_path}!")
        return
        
    logger.info(f"Loading nicely trained final model: {model_path}")
    model = joblib.load(model_path)
    
    # 2. Run predictions for all modern seasons
    all_preds = []
    import traceback
    for s in modern_seasons:
        try:
            preds_df = run_season_predictions(s, model)
            all_preds.append(preds_df)
        except Exception as e:
            logger.error(f"Failed to process season {s}: {e}")
            traceback.print_exc()
            
    if not all_preds:
        logger.error("No predictions could be generated.")
        return
        
    df_all_preds = pd.concat(all_preds, ignore_index=True)
    
    # Ensure analysis directory exists
    analysis_dir = Path(config.ANALYSIS_DIR)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV of raw chronological predictions
    csv_path = analysis_dir / 'poisson_predictive_order_raw.csv'
    df_all_preds.to_csv(csv_path, index=False)
    logger.info(f"Raw chronological predictions saved to {csv_path}")
    
    # 3. Calculate predictive accuracy and binomial uncertainty at each game N
    max_n = 81
    results = []
    
    for n in range(max_n + 1):
        df_n = df_all_preds[df_all_preds['N'] == n]
        if len(df_n) == 0:
            continue
            
        m = len(df_n)
        
        # xG accuracy
        xg_acc = df_n['xg_correct'].mean()
        xg_se = np.sqrt(xg_acc * (1 - xg_acc) / m) if m > 1 else 0.0
        
        # Actual Goals accuracy
        goals_acc = df_n['goals_correct'].mean()
        goals_se = np.sqrt(goals_acc * (1 - goals_acc) / m) if m > 1 else 0.0
        
        results.append({
            'N': n,
            'm': m,
            'xg_acc': xg_acc,
            'xg_se': xg_se,
            'goals_acc': goals_acc,
            'goals_se': goals_se
        })
        
    df_summary = pd.DataFrame(results)
    summary_path = analysis_dir / 'poisson_predictive_order_summary.csv'
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"Summary accuracy stats saved to {summary_path}")
    
    # 4. Generate visual comparison plot
    logger.info("Generating premium visual comparison plot...")
    
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='#f8f9fa')
    ax.set_facecolor('#ffffff')
    
    # Plot individual seasons in lighter colors
    colors_xg = ['#a2d2ff', '#bde0fe', '#ffafcc', '#ffc8dd', '#cdb4db']
    colors_goals = ['#e9c46a', '#f4a261', '#e76f51', '#2a9d8f', '#264653']
    
    for i, s in enumerate(modern_seasons):
        df_s = df_all_preds[df_all_preds['season'] == s]
        s_results = []
        for n in range(max_n + 1):
            df_sn = df_s[df_s['N'] == n]
            if len(df_sn) > 0:
                s_results.append({
                    'N': n,
                    'xg_acc': df_sn['xg_correct'].mean(),
                    'goals_acc': df_sn['goals_correct'].mean()
                })
        df_s_summary = pd.DataFrame(s_results)
        
        # Roll average for smoother individual season visualization
        if len(df_s_summary) > 0:
            ax.plot(df_s_summary['N'] + 1, df_s_summary['xg_acc'].rolling(7, min_periods=1).mean(), 
                    color='#3498db', alpha=0.15, linestyle='--', linewidth=1.2)
            ax.plot(df_s_summary['N'] + 1, df_s_summary['goals_acc'].rolling(7, min_periods=1).mean(), 
                    color='#e74c3c', alpha=0.15, linestyle=':', linewidth=1.2)
            
    # Roll average for aggregate lines to highlight general trend
    df_summary['xg_acc_smooth'] = df_summary['xg_acc'].rolling(7, min_periods=1).mean()
    df_summary['goals_acc_smooth'] = df_summary['goals_acc'].rolling(7, min_periods=1).mean()
    
    # Shaded error bands (95% Binomial Confidence Interval)
    ax.fill_between(df_summary['N'] + 1, 
                    df_summary['xg_acc_smooth'] - 1.96 * df_summary['xg_se'], 
                    df_summary['xg_acc_smooth'] + 1.96 * df_summary['xg_se'], 
                    color='#3498db', alpha=0.12, label='xG Model 95% CI')
                    
    ax.fill_between(df_summary['N'] + 1, 
                    df_summary['goals_acc_smooth'] - 1.96 * df_summary['goals_se'], 
                    df_summary['goals_acc_smooth'] + 1.96 * df_summary['goals_se'], 
                    color='#e74c3c', alpha=0.12, label='Actual Goals 95% CI')
                    
    # Bold aggregate lines
    ax.plot(df_summary['N'] + 1, df_summary['xg_acc_smooth'], 
            color='#1f77b4', linewidth=3, label='xG Model (Aggregate)', marker='o', markevery=10, markersize=6)
    ax.plot(df_summary['N'] + 1, df_summary['goals_acc_smooth'], 
            color='#d62728', linewidth=3, label='Actual Goals Baseline (Aggregate)', marker='s', markevery=10, markersize=6)
            
    # Labels & Title
    ax.set_title("Cronological Predictive Winner Accuracy: xG vs. Actual Goals\nPoisson matchup model evaluated in game-order within seasons (Rolling 7-game average)", 
                 fontsize=15, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_xlabel("N Games of Information Available (Prior Games Played)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("Predictive Winner Accuracy (%)", fontsize=12, fontweight='bold', labelpad=10)
    
    ax.set_xlim(1, 82)
    ax.set_ylim(0.48, 0.65) # standard range for NHL winner predictive limits
    
    ax.grid(True, linestyle=':', alpha=0.6, color='#bdc3c7')
    ax.legend(loc='upper left', facecolor='white', edgecolor='#bdc3c7', framealpha=0.95, fontsize=11)
    
    # Highlight initial coinflip game
    ax.axvline(x=1, color='#7f8c8d', linestyle=':', alpha=0.7)
    ax.annotate("Game 1 (Coinflip)", xy=(1.5, 0.49), xytext=(5, 0.492),
                arrowprops=dict(arrowstyle="->", color='#7f8c8d', lw=0.8), fontsize=10, color='#7f8c8d')
                
    plt.tight_layout()
    
    plot_path = analysis_dir / 'poisson_predictive_order.png'
    fig.savefig(plot_path, dpi=300, facecolor='#f8f9fa')
    plt.close()
    logger.info(f"Visual comparative plot successfully saved to {plot_path}")
    logger.info("Evaluation sweep complete!")

if __name__ == '__main__':
    main()
