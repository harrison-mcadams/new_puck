
import sys
import os
import argparse
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import poisson

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import mixed_effects
from puck.rink import draw_rink

def load_assets(season="20252026"):
    base_dir = Path(f"analysis/mixed_effects_heatmaps_{season}")
    
    # Load Models
    models = {}
    for state in ['5v5', '5v4', '4v5']:
        model_path = base_dir / "models" / f"mixed_model_{state}.pkl"
        if model_path.exists():
            print(f"Loading model: {model_path}")
            models[state] = joblib.load(model_path)
        else:
            print(f"Warning: Model not found {model_path}")
            
    # Load Grids
    grid_path = base_dir / "team_grids.pkl"
    if grid_path.exists():
        print(f"Loading grids: {grid_path}")
        with open(grid_path, 'rb') as f:
            grids = pickle.load(f)
    else:
        grids = {}
        print("Warning: Grids not found.")
        
    # Load Stats Summary (for shot rates)
    stats_path = base_dir / "team_stats_summary.json"
    stats_summary = {}
    if stats_path.exists():
        import json
        with open(stats_path, 'r') as f:
            stats_summary = json.load(f)
    # Load Events Bank for Empirical Sampling
    events_bank_path = base_dir / "events_bank.pkl"
    events_bank = None
    if events_bank_path.exists():
        print(f"Loading events bank: {events_bank_path}")
        events_bank = pd.read_pickle(events_bank_path)
    else:
        print("Warning: Events bank not found (run generate_mixed_heatmaps.py first). Using fallback random.")
            
    return models, grids, stats_summary, events_bank

def get_matchup_density(home_grid_for, away_grid_ag, bins_x, bins_y):
    # Geometric mean of Home Offense and Away Defense
    if home_grid_for.sum() > 0:
        h_pdf = home_grid_for / home_grid_for.sum()
    else: h_pdf = np.zeros_like(home_grid_for)
    
    if away_grid_ag.sum() > 0:
        a_pdf = away_grid_ag / away_grid_ag.sum()
    else: a_pdf = np.zeros_like(away_grid_ag)
    
    # Matchup = sqrt(H * A)
    matchup = np.sqrt(h_pdf * a_pdf)
    
    # Re-normalize
    if matchup.sum() > 0:
        matchup /= matchup.sum()
        
    return matchup

def precalculate_matchup_xg(events_bank, models, home_team, away_team):
    """
    Pre-calculate xG for ALL events in the bank for the specific matchup context.
    This avoids running model inference inside the simulation loop and ensures
    feature consistency.
    """
    print(f"Pre-calculating Matchup xG for {home_team} vs {away_team}...")
    
    # We will add columns: 'xg_home_context', 'xg_away_context'
    events_bank['xg_home_context'] = 0.0
    events_bank['xg_away_context'] = 0.0
    
    for state, model in models.items():
        if model is None: continue
        
        mask = events_bank['game_state'] == state
        if not mask.any(): continue
        
        # Subset for this state
        df_state = events_bank[mask].copy()
        
        # 1. Home Offense Context (Home Team vs Away Team)
        df_home = df_state.copy()
        df_home['team_name'] = home_team
        df_home['opp_team_name'] = away_team
        # Ensure dummy columns for GLM exist (fill 0 if missing in export, shouldn't happen with full export)
        # Predict
        probs_home = model.predict_proba(df_home)[:, 1]
        events_bank.loc[mask, 'xg_home_context'] = probs_home
        
        # 2. Away Offense Context (Away Team vs Home Team)
        df_away = df_state.copy()
        df_away['team_name'] = away_team
        df_away['opp_team_name'] = home_team
        # Predict
        probs_away = model.predict_proba(df_away)[:, 1]
        events_bank.loc[mask, 'xg_away_context'] = probs_away
        
    print("Pre-calculation complete.")
    return events_bank

def sample_team_shots_precalc(events_bank, team_is_home, state, n_shots):
    """
    Sample shots using pre-calculated xG values.
    
    Args:
        team_is_home: Boolean. If True, valid for Home Team (uses xg_home_context).
                      If False, valid for Away Team (uses xg_away_context).
    """
    if events_bank is None or len(events_bank) == 0:
        return None 
        
    # Filter by state
    bank = events_bank[events_bank['game_state'] == state]
    if len(bank) == 0: return None
    
    # Sample Indices
    indices = np.random.choice(bank.index, size=n_shots, replace=True)
    sampled = bank.loc[indices].copy().reset_index(drop=True)
    
    # Assign pre-calculated xG to 'xgs' column
    if team_is_home:
        sampled['xgs'] = sampled['xg_home_context']
    else:
        sampled['xgs'] = sampled['xg_away_context']
    
    return sampled


def _generate_fallback_shots(n_shots, team, opp, state):
    """Generate random shots when no historical data available."""
    # Random spatial distribution (biased toward high danger areas)
    x_vals = np.random.beta(2, 5, size=n_shots) * 40 + 50  # Centered ~65 feet out
    y_vals = np.random.normal(0, 15, size=n_shots).clip(-40, 40)
    
    dist = np.sqrt((89 - x_vals)**2 + y_vals**2)
    dx = np.clip(89 - x_vals, 0.1, None)
    angle = np.abs(np.degrees(np.arctan(y_vals / dx)))
    
    df = pd.DataFrame({
        'x_adj': x_vals,
        'y_adj': y_vals,
        'distance': dist,
        'angle_deg': angle,
        'game_state': state,
        'team_name': team,
        'opp_team_name': opp,
        'home_abb': team,
        'away_abb': opp,
        'is_net_empty': 0,
        'shot_type': np.random.choice(['wrist', 'snap', 'slap', 'backhand'], size=n_shots, p=[0.5, 0.2, 0.2, 0.1]),
        'time_since_last_event': np.random.exponential(15, size=n_shots),
        'is_rebound': np.random.choice([0, 1], size=n_shots, p=[0.85, 0.15]),
        'is_rush': np.random.choice([0, 1], size=n_shots, p=[0.85, 0.15]),
        'shooter_role': np.random.choice(['F', 'D'], size=n_shots, p=[0.7, 0.3]),
        'shoots_catches': np.random.choice(['L', 'R'], size=n_shots, p=[0.6, 0.4]),
        'period': 2,
        'period_seconds': 600,
        'total_time_elapsed_s': 1800,
        'score_diff': 0
    })
    
    df['time_diff'] = df['time_since_last_event']
    df['last_event_time_diff'] = df['time_since_last_event']
    df['period_time_diff'] = df['time_since_last_event']
    
    return df

def simulate_matchup(home, away, models, grids, stats, events_bank=None, n_sims=1000):
    print(f"\n--- Simulating {home} vs {away} (Monte Carlo N={n_sims}) ---")
    
    # Grid Edges
    BIN_X = np.linspace(-100, 100, 201) 
    BIN_Y = np.linspace(-42.5, 42.5, 86)
    
    # --- PREPARE RATES & DENSITIES (Once per matchup) ---
    
    # 1. 5v5
    duration_5v5_min = 48.0
    
    # Home 5v5 Rates
    h_cf60 = stats[home]['5v5']['attempts_for'] / (stats[home]['5v5']['seconds']/3600) if stats[home]['5v5']['seconds'] else 60.0
    a_ca60 = stats[away]['5v5']['attempts_against'] / (stats[away]['5v5']['seconds']/3600) if stats[away]['5v5']['seconds'] else 60.0
    rate_h_5v5 = np.sqrt(h_cf60 * a_ca60) * (duration_5v5_min/60)
    
    dens_h_5v5 = get_matchup_density(grids[home]['5v5']['grid_for'], grids[away]['5v5']['grid_against'], BIN_X, BIN_Y)
    
    # Away 5v5 Rates
    a_cf60 = stats[away]['5v5']['attempts_for'] / (stats[away]['5v5']['seconds']/3600) if stats[away]['5v5']['seconds'] else 60.0
    h_ca60 = stats[home]['5v5']['attempts_against'] / (stats[home]['5v5']['seconds']/3600) if stats[home]['5v5']['seconds'] else 60.0
    rate_a_5v5 = np.sqrt(a_cf60 * h_ca60) * (duration_5v5_min/60)
    
    dens_a_5v5 = get_matchup_density(grids[away]['5v5']['grid_for'], grids[home]['5v5']['grid_against'], BIN_X, BIN_Y)
    
    # 2. Special Teams (Assume 6 mins PP each)
    duration_pp_min = 6.0
    
    # Home PP
    h_pp_cf60 = stats[home]['5v4']['attempts_for'] / (stats[home]['5v4']['seconds']/3600) if stats[home]['5v4']['seconds'] else 60.0
    a_pk_ca60 = stats[away]['4v5']['attempts_against'] / (stats[away]['4v5']['seconds']/3600) if stats[away]['4v5']['seconds'] else 60.0
    rate_h_pp = np.sqrt(h_pp_cf60 * a_pk_ca60) * (duration_pp_min/60)
    
    dens_h_pp = get_matchup_density(grids[home]['5v4']['grid_for'], grids[away]['4v5']['grid_against'], BIN_X, BIN_Y)
    
    # Away PP
    a_pp_cf60 = stats[away]['5v4']['attempts_for'] / (stats[away]['5v4']['seconds']/3600) if stats[away]['5v4']['seconds'] else 60.0
    h_pk_ca60 = stats[home]['4v5']['attempts_against'] / (stats[home]['4v5']['seconds']/3600) if stats[home]['4v5']['seconds'] else 60.0
    rate_a_pp = np.sqrt(a_pp_cf60 * h_pk_ca60) * (duration_pp_min/60)
    
    dens_a_pp = get_matchup_density(grids[away]['5v4']['grid_for'], grids[home]['4v5']['grid_against'], BIN_X, BIN_Y)
    
    print(f"  {home} 5v5: {rate_h_5v5:.1f}, PP: {rate_h_pp:.1f}")
    print(f"  {away} 5v5: {rate_a_5v5:.1f}, PP: {rate_a_pp:.1f}")
    
    # --- PRE-CALCULATION ---
    if events_bank is not None:
        events_bank = precalculate_matchup_xg(events_bank, models, home, away)
    
    # --- SIMULATION LOOP ---
    
    sim_results = {
        'h_goals': [], 'a_goals': [],
        'h_xg': [], 'a_xg': []
    }
    
    import time
    start_time = time.time()
    
    for i in range(n_sims):
        if i > 0 and i % 100 == 0:
            print(f"  Sim {i}/{n_sims}...", end='\r')
            
        # 1. Draw Shot Volumes
        n_h_5v5 = poisson.rvs(rate_h_5v5)
        n_a_5v5 = poisson.rvs(rate_a_5v5)
        n_h_pp = poisson.rvs(rate_h_pp)
        n_a_pp = poisson.rvs(rate_a_pp)
        
        # 2. Generate Shots & Predict
        # Use Helper to handle empty batches gracefully
        def sim_batch_precalc(n, team_is_home, state):
            if n <= 0: return 0.0, 0, 0  # xg, sim_goals, actual_goals
            
            # Sample pre-calculated events
            shots_df = sample_team_shots_precalc(events_bank, team_is_home, state, n)
            
            if shots_df is None or len(shots_df) == 0:
                return 0.0, 0, 0
            
            # Count actual goals from sampled events (for debugging)
            actual_goals = 0
            if 'event' in shots_df.columns:
                actual_goals = (shots_df['event'] == 'goal').sum()
            
            # Use Pre-Calculated xG
            probs = shots_df['xgs'].values
                
            # Sim Outcome (Bernoulli on xG)
            sim_goals = np.sum(np.random.rand(n) < probs)
            xg_sum = np.sum(probs)
            return xg_sum, sim_goals, actual_goals


        xg_h_5, g_h_5, actual_h_5 = sim_batch_precalc(n_h_5v5, True, '5v5')
        xg_a_5, g_a_5, actual_a_5 = sim_batch_precalc(n_a_5v5, False, '5v5')
        
        xg_h_p, g_h_p, actual_h_p = sim_batch_precalc(n_h_pp, True, '5v4')
        xg_a_p, g_a_p, actual_a_p = sim_batch_precalc(n_a_pp, False, '5v4')

        
        # Aggregate
        sim_results['h_goals'].append(g_h_5 + g_h_p)
        sim_results['a_goals'].append(g_a_5 + g_a_p)
        sim_results['h_xg'].append(xg_h_5 + xg_h_p)
        sim_results['a_xg'].append(xg_a_5 + xg_a_p)
        
        # Track breakdown
        if 'h_xg_5v5' not in sim_results:
            sim_results['h_xg_5v5'] = []
            sim_results['h_xg_pp'] = []
            sim_results['a_xg_5v5'] = []
            sim_results['a_xg_pp'] = []
            sim_results['h_actual_5v5'] = []  # Actual goals from sampled events
            sim_results['a_actual_5v5'] = []
            sim_results['h_actual_pp'] = []
            sim_results['a_actual_pp'] = []
        sim_results['h_xg_5v5'].append(xg_h_5)
        sim_results['h_xg_pp'].append(xg_h_p)
        sim_results['a_xg_5v5'].append(xg_a_5)
        sim_results['a_xg_pp'].append(xg_a_p)
        sim_results['h_actual_5v5'].append(actual_h_5)
        sim_results['a_actual_5v5'].append(actual_a_5)
        sim_results['h_actual_pp'].append(actual_h_p)
        sim_results['a_actual_pp'].append(actual_a_p)


    print(f"  Sim 100% Complete ({time.time()-start_time:.1f}s)")
    
    # --- ANALYSIS ---
    h_goals = np.array(sim_results['h_goals'])
    a_goals = np.array(sim_results['a_goals'])
    
    wins_h = np.sum(h_goals > a_goals)
    wins_a = np.sum(a_goals > h_goals)
    ties = np.sum(h_goals == a_goals) 
    
    # OT/Shootout coinflip for ties
    wins_h += ties * 0.5
    wins_a += ties * 0.5
    
    print(f"\n--- Final Results ({n_sims} Games) ---")
    print(f"Win Prob: {home} {100*wins_h/n_sims:.1f}% | {away} {100*wins_a/n_sims:.1f}%")
    print(f"Mean Score: {home} {np.mean(h_goals):.2f} - {away} {np.mean(a_goals):.2f}")
    print(f"Mean xG:    {home} {np.mean(sim_results['h_xg']):.2f} - {away} {np.mean(sim_results['a_xg']):.2f}")
    
    # Show breakdown
    if 'h_xg_5v5' in sim_results:
        print(f"\nxG Breakdown (Mean):")
        print(f"  {home}: 5v5={np.mean(sim_results['h_xg_5v5']):.2f}, PP={np.mean(sim_results['h_xg_pp']):.2f}")
        print(f"  {away}: 5v5={np.mean(sim_results['a_xg_5v5']):.2f}, PP={np.mean(sim_results['a_xg_pp']):.2f}")
        
        if 'h_actual_5v5' in sim_results:
            print(f"\n[DEBUG] Actual Goals from Sampled Events (Mean):")
            print(f"  {home}: 5v5={np.mean(sim_results['h_actual_5v5']):.2f}, PP={np.mean(sim_results['h_actual_pp']):.2f}")
            print(f"  {away}: 5v5={np.mean(sim_results['a_actual_5v5']):.2f}, PP={np.mean(sim_results['a_actual_pp']):.2f}")
            print(f"  (These are goals that actually happened in the sampled historical events)")

    # --- PLOTTING ---
    out_dir = Path(f"analysis/matchups")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histograms
    bins = np.arange(-0.5, max(h_goals.max(), a_goals.max()) + 1.5, 1)
    ax.hist(h_goals, bins=bins, alpha=0.6, label=f"{home} (Mean={np.mean(h_goals):.1f})", color='blue', density=True)
    ax.hist(a_goals, bins=bins, alpha=0.6, label=f"{away} (Mean={np.mean(a_goals):.1f})", color='orange', density=True)
    
    ax.set_title(f"Simulated Score Distribution: {home} vs {away}")
    ax.set_xlabel("Goals Scored")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    out_path = out_dir / f"sim_dist_{home}_{away}.png"
    plt.savefig(out_path)
    print(f"Saved distribution plot to {out_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("home", help="Home Team Abbr (e.g. NYR)")
    parser.add_argument("away", help="Away Team Abbr (e.g. NJD)")
    args = parser.parse_args()
    
    models, grids, stats, events_bank = load_assets()
    if not models:
        print("No models loaded. Run generate_mixed_heatmaps.py first.")
        return
        
    simulate_matchup(args.home, args.away, models, grids, stats, events_bank)

if __name__ == "__main__":
    main()
