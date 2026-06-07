"""scripts/predict_baseline.py

Chronological Poisson validation routine to compare xG vs actual goals
under 5v5 vs all situations, utilizing flat-50 and skill-biased tie-breakers.
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
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from puck import config, analyze, data_pipeline
from scripts.evaluate_predictive_power import DataUtils

def load_season_data_fast(season):
    logger.info(f"Loading modeled data for {season}...")
    csv_path = os.path.join('data', str(season), f"{season}_df_modeled.csv")
    df = pd.read_csv(csv_path)
    return df

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Premium Styling Settings
try:
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
except:
    pass
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Outfit', 'DejaVu Sans', 'Arial']

def calculate_poisson_win_prob(h_exp, a_exp, tie_breaker='flat_50'):
    """
    Calculates Home win probability using independent Poisson distributions.
    Resolves regulation ties based on selected strategy.
    """
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
        
    if tie_breaker == 'flat_50':
        p_tie_break = 0.5
    elif tie_breaker == 'skill_biased':
        # Proportional to expected goals
        p_tie_break = h_exp / (h_exp + a_exp)
    else:
        p_tie_break = 0.5
        
    return home_wins + p_tie_break * ties

def add_shooting_team(df):
    """
    Correctly attributes shooting and defending teams.
    Handles blocked shots where event team is defender.
    """
    df = df.copy()
    is_home_event = df['team_id'].astype(str) == df['home_id'].astype(str)
    
    df['shooting_team_abb'] = np.where(is_home_event, df['home_abb'], df['away_abb'])
    df['defending_team_abb'] = np.where(is_home_event, df['away_abb'], df['home_abb'])
    
    # Blocked shots: event team is the blocker (defender)
    mask_blocked = df['event'] == 'blocked-shot'
    mask_home_block = mask_blocked & is_home_event
    mask_away_block = mask_blocked & ~is_home_event
    
    df.loc[mask_home_block, 'shooting_team_abb'] = df.loc[mask_home_block, 'away_abb']
    df.loc[mask_home_block, 'defending_team_abb'] = df.loc[mask_home_block, 'home_abb']
    
    df.loc[mask_away_block, 'shooting_team_abb'] = df.loc[mask_away_block, 'home_abb']
    df.loc[mask_away_block, 'defending_team_abb'] = df.loc[mask_away_block, 'away_abb']
    
    return df

preprocessed_cache = {}
scored_cache = {}

def get_preprocessed_season_data(season):
    if season in preprocessed_cache:
        logger.info(f"  [CACHE HIT] Using cached preprocessed data for Season {season}")
        return preprocessed_cache[season]
    
    df = load_season_data_fast(season)
    df['season'] = int(season)
    df = add_shooting_team(df)
    preprocessed_cache[season] = df
    return df

def get_scored_season_data(season, model, metric, situation):
    cache_key = (season, metric, situation)
    if cache_key in scored_cache:
        logger.info(f"  [CACHE HIT] Using cached scored/filtered data for Season {season} | Metric: {metric} | Situation: {situation}")
        return scored_cache[cache_key]
    
    # 1. Get preprocessed data
    df = get_preprocessed_season_data(season)
    
    # 2. Score
    df_scored = df.copy()
    if metric == 'xg':
        logger.info(f"  [CACHE MISS] Loading pre-modeled expected goals (xgs) for Season {season}...")
        if 'xgs' not in df_scored.columns:
            logger.error(f"  Column 'xgs' not found in Season {season} modeled data! Available: {df_scored.columns.tolist()}")
            raise ValueError(f"'xgs' column missing for season {season}")
        df_scored['eval_val'] = df_scored['xgs']
    else:
        logger.info(f"  [CACHE MISS] Mapping actual goals for Season {season}...")
        df_scored['eval_val'] = (df_scored['event'].str.lower() == 'goal').astype(float)
        
    # Apply situation filter
    if situation == '5v5':
        df_filtered = df_scored[df_scored['game_state'] == '5v5'].copy()
    else:
        df_filtered = df_scored.copy()
        
    scored_cache[cache_key] = df_filtered
    return df_filtered

def run_season_chronological(season, model, metric='xg', situation='all', tie_breaker='flat_50', min_games=5):
    """
    Runs chronological evaluation for a single season under one configuration.
    """
    logger.info(f"Running Season {season} | Metric: {metric} | Situation: {situation} | Tie-Breaker: {tie_breaker}...")
    
    # Get preprocessed and scored/filtered data (cached)
    df = get_preprocessed_season_data(season)
    df_filtered = get_scored_season_data(season, model, metric, situation)
    
    # Reconstruct schedule
    sched_df = DataUtils.process_schedule(df)
    sched_df = sched_df.sort_values('game_id').reset_index(drop=True)
    
    # Pre-calculate game-level totals to speed up chronological loop
    game_totals = {}
    grouped = df_filtered.groupby(['game_id', 'shooting_team_abb'])['eval_val'].sum().reset_index()
    for _, row in grouped.iterrows():
        gid = row['game_id']
        team = row['shooting_team_abb']
        val = row['eval_val']
        if gid not in game_totals:
            game_totals[gid] = {}
        game_totals[gid][team] = val
        
    # Track team history: stats accumulated so far
    # keys: 'for_sum', 'against_sum', 'games_played'
    teams = pd.concat([sched_df['home_team'], sched_df['away_team']]).unique()
    team_stats = {t: {'for_sum': 0.0, 'against_sum': 0.0, 'games_played': 0} for t in teams}
    
    predictions = []
    
    for _, row in sched_df.iterrows():
        gid = row['game_id']
        h, a = row['home_team'], row['away_team']
        
        h_goals = row['home_goals_final']
        a_goals = row['away_goals_final']
        
        # Get actual results in this game for situation updating
        h_val = game_totals.get(gid, {}).get(h, 0.0)
        a_val = game_totals.get(gid, {}).get(a, 0.0)
        
        m_H = team_stats[h]['games_played']
        m_A = team_stats[a]['games_played']
        
        # Prior games info available (average)
        N = int(np.round((m_H + m_A) / 2))
        
        # Predict winner probability
        if m_H < min_games or m_A < min_games:
            p_win_home = 0.5
        else:
            # Rates per game
            off_H = team_stats[h]['for_sum'] / m_H
            def_H = team_stats[h]['against_sum'] / m_H
            off_A = team_stats[a]['for_sum'] / m_A
            def_A = team_stats[a]['against_sum'] / m_A
            
            # League average rate so far
            total_for = sum(t['for_sum'] for t in team_stats.values())
            total_games = sum(t['games_played'] for t in team_stats.values())
            # Each game counts as 2 team-games of play
            league_avg = total_for / total_games if total_games > 0 else 3.0
            
            # Matchup expectations
            lambda_H = (off_H * def_A) / league_avg if league_avg > 0 else 0.0
            lambda_A = (off_A * def_H) / league_avg if league_avg > 0 else 0.0
            
            p_win_home = calculate_poisson_win_prob(lambda_H, lambda_A, tie_breaker=tie_breaker)
            
        # Actual outcome
        actual_winner = 1.0 if h_goals > a_goals else 0.0
        
        # Accuracy
        if p_win_home > 0.5:
            correct = 1.0 if actual_winner == 1.0 else 0.0
        elif p_win_home < 0.5:
            correct = 1.0 if actual_winner == 0.0 else 0.0
        else:
            correct = 0.5
            
        brier = (actual_winner - p_win_home) ** 2
        
        predictions.append({
            'season': season,
            'game_id': gid,
            'N': N,
            'p_home': p_win_home,
            'y_actual': actual_winner,
            'correct': correct,
            'brier': brier
        })
        
        # Update team stats with this game's filtered outputs
        team_stats[h]['for_sum'] += h_val
        team_stats[h]['against_sum'] += a_val
        team_stats[h]['games_played'] += 1
        
        team_stats[a]['for_sum'] += a_val
        team_stats[a]['against_sum'] += h_val
        team_stats[a]['games_played'] += 1
        
    return pd.DataFrame(predictions)

def plot_fitted_trendline(ax, x, y, label, color, linestyle='-', smooth_window=7):
    """
    Plots the rolling average as a smooth line, raw points (lightly), 
    and a fitted lowess or polynomial trendline.
    """
    # 1. Roll average
    df_temp = pd.DataFrame({'x': x, 'y': y}).sort_values('x')
    rolling_y = df_temp['y'].rolling(smooth_window, min_periods=1, center=True).mean()
    ax.plot(df_temp['x'], rolling_y, color=color, linewidth=2.5, label=f"{label} (Rolling {smooth_window})", linestyle=linestyle)
    
    # 2. Scatter raw points (lightly)
    ax.scatter(df_temp['x'], df_temp['y'], color=color, alpha=0.1, s=15, label='_nolegend_')
    
    # 3. Fit and plot trendline (LOWESS if statsmodels available, else polynomial)
    try:
        from statsmodels.api import nonparametric
        lowess_fit = nonparametric.lowess(df_temp['y'], df_temp['x'], frac=0.4)
        ax.plot(lowess_fit[:, 0], lowess_fit[:, 1], color=color, linewidth=1.5, linestyle='--', alpha=0.8, label=f"{label} (LOWESS)")
    except ImportError:
        # Fallback to 3rd degree polynomial fit
        try:
            poly_coefs = np.polyfit(df_temp['x'], df_temp['y'], deg=3)
            poly_fit = np.poly1d(poly_coefs)
            ax.plot(df_temp['x'], poly_fit(df_temp['x']), color=color, linewidth=1.5, linestyle='--', alpha=0.8, label=f"{label} (Poly Fit)")
        except Exception as e:
            logger.warning(f"Could not fit trendline: {e}")

def main():
    logger.info("=================================================================")
    logger.info("ITERATION 3: POISSON MATCHUP BASELINE BACKTEST Sweep")
    logger.info("=================================================================")
    
    # Discover available seasons
    all_seasons = DataUtils.get_available_seasons()
    seasons_to_run = sorted([s for s in all_seasons if os.path.exists(os.path.join('data', str(s), f"{s}_df_modeled.csv"))])
    model = None
    
    # Define configurations to compare
    # Format: (metric, situation, label, color)
    configs = [
        ('xg', 'all', 'xG Model (All)', '#1f77b4'),      # Blue
        ('xg', '5v5', 'xG Model (5v5)', '#3498db'),      # Light Blue
        ('actual', 'all', 'Actual Goals (All)', '#d62728'), # Red
        ('actual', '5v5', 'Actual Goals (5v5)', '#e74c3c')  # Light Red
    ]
    
    tie_breakers = ['skill_biased']
    
    # Dictionary to hold all raw prediction DFs
    # key: (tie_breaker, metric, situation)
    results = {}
    
    for tb in tie_breakers:
        for metric, situation, label, color in configs:
            all_preds = []
            for season in seasons_to_run:
                try:
                    preds_df = run_season_chronological(season, model, metric=metric, situation=situation, tie_breaker=tb)
                    all_preds.append(preds_df)
                except Exception as e:
                    logger.error(f"Failed to process season {season} for config {label}/{tb}: {e}")
                    import traceback
                    traceback.print_exc()
            
            if all_preds:
                df_all = pd.concat(all_preds, ignore_index=True)
                results[(tb, metric, situation)] = df_all
                
    # Save CSV of raw prediction logs for audit
    analysis_dir = Path(config.ANALYSIS_DIR) / 'evaluation'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare comparative summary statistics
    # Aggregate stats at each game N
    max_n = 81
    
    for tb in tie_breakers:
        summary_rows = []
        for metric, situation, label, color in configs:
            df_config = results.get((tb, metric, situation))
            if df_config is None: continue
            
            for n in range(max_n + 1):
                df_n = df_config[df_config['N'] == n]
                if len(df_n) == 0: continue
                
                acc = df_n['correct'].mean()
                brier = df_n['brier'].mean()
                summary_rows.append({
                    'Tie_Breaker': tb,
                    'Metric': metric,
                    'Situation': situation,
                    'Label': label,
                    'N': n,
                    'Games': len(df_n),
                    'Accuracy': acc,
                    'Brier': brier
                })
        
        df_summary = pd.DataFrame(summary_rows)
        summary_path = analysis_dir / f'predict_baseline_summary_{tb}.csv'
        df_summary.to_csv(summary_path, index=False)
        logger.info(f"Summary saved to {summary_path}")
        
    # --- SEASON-BY-SEASON SUMMARY STATISTICS ---
    logger.info("Computing season-by-season statistics...")
    season_results = []
    for tb in tie_breakers:
        for metric, situation, label, color in configs:
            df_config = results.get((tb, metric, situation))
            if df_config is None: continue
            
            # Group by season
            grouped = df_config.groupby('season')
            for season, group in grouped:
                acc = group['correct'].mean()
                brier = group['brier'].mean()
                season_results.append({
                    'Tie_Breaker': tb,
                    'Metric': metric,
                    'Situation': situation,
                    'Label': label,
                    'Season': int(season),
                    'Games': len(group),
                    'Accuracy': acc,
                    'Brier': brier
                })
                
    df_seasons_summary = pd.DataFrame(season_results)
    seasons_summary_path = analysis_dir / 'predict_baseline_seasons_summary.csv'
    df_seasons_summary.to_csv(seasons_summary_path, index=False)
    logger.info(f"Season summary saved to {seasons_summary_path}")
        
    # --- VISUALIZATION GENERATION (PLOT 1: Curves vs N) ---
    logger.info("Generating curves vs N plots (Skill-Biased Only)...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), facecolor='#f8f9fa')
    
    for col_idx, metric_type in enumerate(['Accuracy', 'Brier']):
        ax = axes[col_idx]
        ax.set_facecolor('#ffffff')
        
        title_suffix = "Accuracy (Higher is Better)" if metric_type == 'Accuracy' else "Brier Score (Lower is Better)"
        
        ax.set_title(f"{title_suffix} vs. Games Played\n(Skill-Biased Proportional OT Split)", fontsize=14, fontweight='bold', color='#2c3e50')
        ax.set_xlabel("N Games of Information Available (Prior Games Played)", fontsize=11, fontweight='bold')
        ax.set_ylabel(f"Predictive {metric_type}", fontsize=11, fontweight='bold')
        
        # Load summary data
        summary_path = analysis_dir / 'predict_baseline_summary_skill_biased.csv'
        if not summary_path.exists(): continue
        df_summary = pd.read_csv(summary_path)
        
        for metric, situation, label, color in configs:
            df_sub = df_summary[(df_summary['Metric'] == metric) & (df_summary['Situation'] == situation)]
            if df_sub.empty: continue
            
            y_vals = df_sub['Accuracy'] if metric_type == 'Accuracy' else df_sub['Brier']
            plot_fitted_trendline(
                ax, df_sub['N'].values, y_vals.values, 
                label=label, color=color, 
                linestyle='-' if 'xG' in label else ':'
            )
            
        ax.set_xlim(1, 82)
        if metric_type == 'Accuracy':
            ax.set_ylim(0.48, 0.63)
        else:
            ax.set_ylim(0.20, 0.26)
            
        ax.grid(True, linestyle=':', alpha=0.6, color='#bdc3c7')
        ax.legend(loc='upper left' if metric_type == 'Accuracy' else 'upper right', 
                  facecolor='white', edgecolor='#bdc3c7', framealpha=0.95, fontsize=10)
        
        # Initial coinflip game reference line
        ax.axvline(x=1, color='#7f8c8d', linestyle=':', alpha=0.7)
            
    plt.tight_layout()
    plot_path = analysis_dir / 'predict_baseline_curves.png'
    fig.savefig(plot_path, dpi=300, facecolor='#f8f9fa')
    plt.close()
    logger.info(f"Curves vs N plots saved to {plot_path}")
    
    # --- VISUALIZATION GENERATION (PLOT 2: Season-by-Season Performance) ---
    logger.info("Generating season-by-season line plots (Skill-Biased Only)...")
    fig_seasons, axes_seasons = plt.subplots(1, 2, figsize=(20, 8), facecolor='#f8f9fa')
    
    for col_idx, metric_type in enumerate(['Accuracy', 'Brier']):
        ax = axes_seasons[col_idx]
        ax.set_facecolor('#ffffff')
        
        title_suffix = "Accuracy (Higher is Better)" if metric_type == 'Accuracy' else "Brier Score (Lower is Better)"
        
        ax.set_title(f"Season-by-Season {title_suffix}\n(Skill-Biased Proportional OT Split)", fontsize=14, fontweight='bold', color='#2c3e50')
        ax.set_xlabel("Season", fontsize=11, fontweight='bold')
        ax.set_ylabel(f"Predictive {metric_type}", fontsize=11, fontweight='bold')
        
        df_sub_summary = df_seasons_summary[df_seasons_summary['Tie_Breaker'] == 'skill_biased']
        
        # Pre-calculate x tick positions and labels from sorted seasons
        all_summary_seasons = sorted(df_seasons_summary['Season'].unique())
        seasons_labels = []
        for s in all_summary_seasons:
            s_str = str(s)
            seasons_labels.append(f"{s_str[:4]}-{s_str[6:]}")
        
        for metric, situation, label, color in configs:
            df_sub = df_sub_summary[(df_sub_summary['Metric'] == metric) & (df_sub_summary['Situation'] == situation)]
            if df_sub.empty: continue
            
            df_sub = df_sub.sort_values('Season')
            
            # Align values with sorted seasons
            y_vals = []
            for s in all_summary_seasons:
                row_s = df_sub[df_sub['Season'] == s]
                if not row_s.empty:
                    y_vals.append(row_s['Accuracy'].values[0] if metric_type == 'Accuracy' else row_s['Brier'].values[0])
                else:
                    y_vals.append(np.nan)
                    
            ax.plot(range(len(all_summary_seasons)), y_vals, color=color, marker='o', linewidth=2.5, label=label,
                    linestyle='-' if 'xG' in label else ':')
            
        ax.grid(True, linestyle=':', alpha=0.6, color='#bdc3c7')
        ax.set_xticks(range(len(all_summary_seasons)))
        ax.set_xticklabels(seasons_labels, rotation=45, ha='right')
        ax.legend(loc='best', facecolor='white', edgecolor='#bdc3c7', framealpha=0.95, fontsize=10)
            
    plt.tight_layout()
    seasons_plot_path = analysis_dir / 'predict_baseline_seasons.png'
    fig_seasons.savefig(seasons_plot_path, dpi=300, facecolor='#f8f9fa')
    plt.close()
    logger.info(f"Season-by-season plots saved to {seasons_plot_path}")
    logger.info("Baseline backtest sweep complete!")

if __name__ == '__main__':
    main()
