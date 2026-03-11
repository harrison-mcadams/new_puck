
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
    
    # Load Unified Mixed Effects Model
    model = None
    model_path = base_dir / "models" / "joint_mixed_effects.joblib"
    # Actually, v2 model is saved at analysis/xgs/joint_mixed_effects.joblib
    # Let's check both paths to be safe, but default to the known good one
    global_model_path = Path("analysis/xgs/joint_mixed_effects.joblib")
    
    if global_model_path.exists():
        print(f"Loading global model: {global_model_path}")
        model = joblib.load(global_model_path)
    elif model_path.exists():
        print(f"Loading local model: {model_path}")
        model = joblib.load(model_path)
    else:
        print(f"Error: Unified model not found at {global_model_path} or {model_path}")
        # Return empty dictionary dict so existing code doesn't crash completely during iteration if it has fallbacks
        model = {}
            
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
            
    return model, grids, stats_summary, events_bank

def get_matchup_density(home_grid_for, away_grid_ag, bins_x, bins_y):
    # Geometric mean of Home Offense and Away Defense
    if home_grid_for.sum() > 0:
        h_pdf = home_grid_for / home_grid_for.sum()
    else: 
        print("DEBUG: Home Grid Sum is 0!")
        h_pdf = np.zeros_like(home_grid_for)
    
    if away_grid_ag.sum() > 0:
        a_pdf = away_grid_ag / away_grid_ag.sum()
    else: 
        print("DEBUG: Away Grid Sum is 0!")
        a_pdf = np.zeros_like(away_grid_ag)
    
    # Matchup = sqrt(H * A)
    matchup = np.sqrt(h_pdf * a_pdf)
    
    if matchup.sum() == 0:
        print(f"DEBUG: Matchup resulted in 0. Overlap Sum: {np.sum(h_pdf * a_pdf)}")
    
    # Re-normalize
    if matchup.sum() > 0:
        matchup /= matchup.sum()
        
    return matchup

def precalculate_matchup_xg(events_bank, model, home_team, away_team):
    """
    Pre-calculate xG for ALL events in the bank for the specific matchup context.
    This avoids running model inference inside the simulation loop and ensures
    feature consistency.
    """
    print(f"Pre-calculating Matchup xG for {home_team} vs {away_team}...")
    
    # We will add columns: 'xg_home_context', 'xg_away_context'
    events_bank['xg_home_context'] = 0.0
    events_bank['xg_away_context'] = 0.0
    events_bank['xg_league_context'] = 0.0
    
    for state in ['5v5', '5v4', '4v5']:
        mask = events_bank['game_state'] == state
        if not mask.any(): continue
        
        # Subset for this state
        df_state = events_bank[mask].copy()
        
        # 1. Home Offense Context (Home Team vs Away Team)
        df_home = df_state.copy()
        df_home['off_team_name'] = home_team
        df_home['def_team_name'] = away_team
        df_home['is_home'] = 1.0
        df_home['relative_game_state'] = state
        # Force all numerics to float64 to avoid imputer type errors
        num_cols = df_home.select_dtypes(include=['int64', 'int32']).columns
        if len(num_cols) > 0: df_home[num_cols] = df_home[num_cols].astype('float64')
        probs_home = model.predict_proba(df_home)[:, 1]
        events_bank.loc[mask, 'xg_home_context'] = probs_home
        
        # 2. Away Offense Context (Away Team vs Home Team)
        df_away = df_state.copy()
        df_away['off_team_name'] = away_team
        df_away['def_team_name'] = home_team
        df_away['is_home'] = 0.0
        if state == '5v4':
            df_away['relative_game_state'] = '4v5'
        elif state == '4v5':
            df_away['relative_game_state'] = '5v4'
        else:
            df_away['relative_game_state'] = state
        num_cols = df_away.select_dtypes(include=['int64', 'int32']).columns
        if len(num_cols) > 0: df_away[num_cols] = df_away[num_cols].astype('float64')
        probs_away = model.predict_proba(df_away)[:, 1]
        events_bank.loc[mask, 'xg_away_context'] = probs_away
        
        # 3. League Context (Average vs Average)
        df_league = df_state.copy()
        df_league['off_team_name'] = 'Average'
        df_league['def_team_name'] = 'Average'
        df_league['is_home'] = 0.5 # Neutral site or average home/away
        
        # For League Context, use the relative_game_state already computed by data_pipeline
        # This correctly maps the state from the shooting team's perspective
        if 'relative_game_state' in df_state.columns:
            df_league['relative_game_state'] = df_state['relative_game_state'].values
        else:
            df_league['relative_game_state'] = state
            
        num_cols = df_league.select_dtypes(include=['int64', 'int32']).columns
        if len(num_cols) > 0: df_league[num_cols] = df_league[num_cols].astype('float64')
        probs_league = model.predict_proba(df_league)[:, 1]
        events_bank.loc[mask, 'xg_league_context'] = probs_league
        
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

def simulate_matchup(home, away, season, n_sims=1000):
    print(f"\n--- Simulating {home} vs {away} (Monte Carlo N={n_sims}) ---")
    
    # 1. Load Assets
    print("Loading empirical data and models...")
    model, grids, stats, events_bank = load_assets(season)
    
    # --- PRE-CALCULATION ---
    if events_bank is not None:
        events_bank = precalculate_matchup_xg(events_bank, model, home, away)
    else:
        print("ERROR: Events bank is required for density calculation now.")
        return

    # Grid Edges
    BIN_X = np.linspace(-100, 100, 201) 
    BIN_Y = np.linspace(-42.5, 42.5, 86)
    
    # --- PREPARE RATES & DENSITIES (Once per matchup) ---
    
    duration_5v5_min = 48.0
    duration_pp_min = 6.0
    
    # Home 5v5 Rates
    h_cf60 = stats[home]['5v5']['attempts_for'] / (stats[home]['5v5']['seconds']/3600) if stats[home]['5v5']['seconds'] else 60.0
    a_ca60 = stats[away]['5v5']['attempts_against'] / (stats[away]['5v5']['seconds']/3600) if stats[away]['5v5']['seconds'] else 60.0
    rate_h_5v5 = np.sqrt(h_cf60 * a_ca60) * (duration_5v5_min/60)
    
    # Away 5v5 Rates
    a_cf60 = stats[away]['5v5']['attempts_for'] / (stats[away]['5v5']['seconds']/3600) if stats[away]['5v5']['seconds'] else 60.0
    h_ca60 = stats[home]['5v5']['attempts_against'] / (stats[home]['5v5']['seconds']/3600) if stats[home]['5v5']['seconds'] else 60.0
    rate_a_5v5 = np.sqrt(a_cf60 * h_ca60) * (duration_5v5_min/60)
    
    # Home PP
    h_pp_cf60 = stats[home]['5v4']['attempts_for'] / (stats[home]['5v4']['seconds']/3600) if stats[home]['5v4']['seconds'] else 60.0
    a_pk_ca60 = stats[away]['4v5']['attempts_against'] / (stats[away]['4v5']['seconds']/3600) if stats[away]['4v5']['seconds'] else 60.0
    rate_h_pp = np.sqrt(h_pp_cf60 * a_pk_ca60) * (duration_pp_min/60)
    
    # Away PP
    a_pp_cf60 = stats[away]['5v4']['attempts_for'] / (stats[away]['5v4']['seconds']/3600) if stats[away]['5v4']['seconds'] else 60.0
    h_pk_ca60 = stats[home]['4v5']['attempts_against'] / (stats[home]['4v5']['seconds']/3600) if stats[home]['4v5']['seconds'] else 60.0
    rate_a_pp = np.sqrt(a_pp_cf60 * h_pk_ca60) * (duration_pp_min/60)
    
    print(f"  {home} 5v5: {rate_h_5v5:.1f}, PP: {rate_h_pp:.1f}")
    print(f"  {away} 5v5: {rate_a_5v5:.1f}, PP: {rate_a_pp:.1f}")
    
    # Densities from Events Bank (5v5)
    print("  Generating High-Fidelity Matchup Density (5v5)...")
    mask_5v5 = events_bank['game_state'] == '5v5'
    
    # We want Home Density = Shots BY Home + Shots AGAINST Away
    # We want Away Density = Shots BY Away + Shots AGAINST Home
    
    mask_h_for = (events_bank['team_name'] == home)
    mask_a_ag = (events_bank['opp_team_name'] == away)
    mask_home_density = mask_5v5 & (mask_h_for | mask_a_ag)
    
    mask_a_for = (events_bank['team_name'] == away)
    mask_h_ag = (events_bank['opp_team_name'] == home)
    mask_away_density = mask_5v5 & (mask_a_for | mask_h_ag)
    
    # 1. Total League Seconds (Scale to per 60)
    # The events bank contains all shots. To scale to a "per pixel per 60 minutes" rate,
    # we need the total 5v5 seconds played by all teams.
    # We can get this from the stats summary.
    total_league_seconds_5v5 = sum(t.get('5v5', {}).get('seconds', 0) for t in stats.values())
    if total_league_seconds_5v5 <= 0:
        print("ERROR: Could not calculate total league seconds for 5v5 from stats summary.")
        return
        
    scale_factor_league = 3600.0 / total_league_seconds_5v5
    
    # We also need Team Seconds to scale the team-specific densities correctly
    # Home Density uses events from Home games OR Away games.
    # We should scale it by the combined 5v5 seconds of Home + Away (approx 2 team-games worth).
    # But wait, Home Offense + Away Defense is 2 'team-seasons' worth of shots.
    # So we want to divide by (Home Seconds + Away Seconds).
    home_sec = stats[home]['5v5']['seconds'] if stats[home]['5v5']['seconds'] else 1.0
    away_sec = stats[away]['5v5']['seconds'] if stats[away]['5v5']['seconds'] else 1.0
    scale_factor_matchup = 3600.0 / (home_sec + away_sec)
    
    # Home Density
    x_h = events_bank.loc[mask_home_density, 'x_adj'].values
    y_h = events_bank.loc[mask_home_density, 'y_adj'].values
    w_h_matchup = events_bank.loc[mask_home_density, 'xg_home_context'].values
    H_home, _, _ = np.histogram2d(x_h, y_h, bins=[BIN_X, BIN_Y], weights=w_h_matchup)
    dens_h_raw = H_home.T * scale_factor_matchup * 100.0 
    
    # Away Density
    x_a = events_bank.loc[mask_away_density, 'x_adj'].values
    y_a = events_bank.loc[mask_away_density, 'y_adj'].values
    w_a_matchup = events_bank.loc[mask_away_density, 'xg_away_context'].values
    H_away, _, _ = np.histogram2d(x_a, y_a, bins=[BIN_X, BIN_Y], weights=w_a_matchup)
    dens_a_raw = H_away.T * scale_factor_matchup * 100.0
    
    # League Density (All 5v5 Shots)
    x_l = events_bank.loc[mask_5v5, 'x_adj'].values
    y_l = events_bank.loc[mask_5v5, 'y_adj'].values
    w_l_base = events_bank.loc[mask_5v5, 'xg_league_context'].values
    H_league, _, _ = np.histogram2d(x_l, y_l, bins=[BIN_X, BIN_Y], weights=w_l_base)
    dens_l_raw = H_league.T * scale_factor_league * 100.0
    
    # Convert to Rate per 60 per pixel
    # Then subtract the League Average Rate per pixel to get the Difference Map
    dens_h_5v5 = dens_h_raw - dens_l_raw
    dens_a_5v5 = dens_a_raw - dens_l_raw
    
    # --- Densities from Events Bank (PP) ---
    print("  Generating High-Fidelity Matchup Density (PP)...")
    
    # League PP Time (5v4 represents all PP time because 1 team is always 5v4)
    total_league_seconds_pp = sum(t.get('5v4', {}).get('seconds', 0) for t in stats.values())
    if total_league_seconds_pp <= 0: total_league_seconds_pp = 1.0
    scale_factor_league_pp = 3600.0 / total_league_seconds_pp
    
    # Team PP Time Setup
    home_pp_sec = stats[home]['5v4']['seconds'] if stats[home]['5v4']['seconds'] else 1.0
    away_pk_sec = stats[away]['4v5']['seconds'] if stats[away]['4v5']['seconds'] else 1.0
    scale_factor_matchup_h_pp = 3600.0 / (home_pp_sec + away_pk_sec)
    
    away_pp_sec = stats[away]['5v4']['seconds'] if stats[away]['5v4']['seconds'] else 1.0
    home_pk_sec = stats[home]['4v5']['seconds'] if stats[home]['4v5']['seconds'] else 1.0
    scale_factor_matchup_a_pp = 3600.0 / (away_pp_sec + home_pk_sec)
    
    # Masks for Home PP Density (Home shooting on PP + Away defending on PK)
    mask_home_pp_density = ((events_bank['team_name'] == home) & (events_bank['game_state'] == '5v4')) | \
                           ((events_bank['opp_team_name'] == away) & (events_bank['game_state'] == '5v4'))
                           
    # Masks for Away PP Density (Away shooting on PP + Home defending on PK)
    mask_away_pp_density = ((events_bank['team_name'] == away) & (events_bank['game_state'] == '5v4')) | \
                           ((events_bank['opp_team_name'] == home) & (events_bank['game_state'] == '5v4'))
                           
    # Masks for League PP Density (Any team shooting on PP)
    mask_league_pp_shots = (events_bank['game_state'] == '5v4')

    # Calculate Home PP Density
    x_h_pp = events_bank.loc[mask_home_pp_density, 'x_adj'].values
    y_h_pp = events_bank.loc[mask_home_pp_density, 'y_adj'].values
    w_h_pp = events_bank.loc[mask_home_pp_density, 'xg_home_context'].values
    H_h_pp, _, _ = np.histogram2d(x_h_pp, y_h_pp, bins=[BIN_X, BIN_Y], weights=w_h_pp)
    dens_h_pp_raw = H_h_pp.T * scale_factor_matchup_h_pp * 100.0
    
    # Calculate Away PP Density
    x_a_pp = events_bank.loc[mask_away_pp_density, 'x_adj'].values
    y_a_pp = events_bank.loc[mask_away_pp_density, 'y_adj'].values
    w_a_pp = events_bank.loc[mask_away_pp_density, 'xg_away_context'].values
    H_a_pp, _, _ = np.histogram2d(x_a_pp, y_a_pp, bins=[BIN_X, BIN_Y], weights=w_a_pp)
    dens_a_pp_raw = H_a_pp.T * scale_factor_matchup_a_pp * 100.0
    
    # Calculate League Base PP Density
    x_l_pp = events_bank.loc[mask_league_pp_shots, 'x_adj'].values
    y_l_pp = events_bank.loc[mask_league_pp_shots, 'y_adj'].values
    w_l_pp = events_bank.loc[mask_league_pp_shots, 'xg_league_context'].values
    H_l_pp, _, _ = np.histogram2d(x_l_pp, y_l_pp, bins=[BIN_X, BIN_Y], weights=w_l_pp)
    dens_l_pp_raw = H_l_pp.T * scale_factor_league_pp * 100.0
    
    dens_h_pp = dens_h_pp_raw - dens_l_pp_raw
    dens_a_pp = dens_a_pp_raw - dens_l_pp_raw

    # We no longer need the league dens passing into the plotter because the grids are already differences
    league_dens_5v5 = None
    
    # Store density components for the dashboard
    density_stats = {
        'h_matchup_xg': w_h_matchup.sum(),
        'a_matchup_xg': w_a_matchup.sum(),
        'h_matchup_sec': home_sec + away_sec,
        'a_matchup_sec': home_sec + away_sec,
        'l_base_xg': w_l_base.sum(),
        'l_base_sec': total_league_seconds_5v5,
        
        'h_pp_matchup_xg': w_h_pp.sum(),
        'a_pp_matchup_xg': w_a_pp.sum(),
        'h_pp_matchup_sec': home_pp_sec + away_pk_sec,
        'a_pp_matchup_sec': away_pp_sec + home_pk_sec,
    }
    
    # --- SIMULATION ARRAYS PRE-EXTRACTION ---
    # Restrict shot sampling only to relevant events (Team A For + Team B Against)
    # This also massively speeds up the simulation loop by avoiding .loc inside
    
    def get_safe_mask(mask, fallback_mask):
        return mask if mask.sum() > 0 else fallback_mask
        
    mask_h_5v5_sample = get_safe_mask(mask_home_density, mask_5v5)
    mask_a_5v5_sample = get_safe_mask(mask_away_density, mask_5v5)
    
    # Notice mask_5v4 is used as fallback for both home PP and away PP (since away PP is also 5v4)
    mask_5v4 = events_bank['game_state'] == '5v4'
    mask_h_pp_sample = get_safe_mask(mask_home_pp_density, mask_5v4)
    mask_a_pp_sample = get_safe_mask(mask_away_pp_density, mask_5v4)
    
    pool_h_5v5 = events_bank.loc[mask_h_5v5_sample, 'xg_home_context'].values
    pool_a_5v5 = events_bank.loc[mask_a_5v5_sample, 'xg_away_context'].values
    pool_h_pp = events_bank.loc[mask_h_pp_sample, 'xg_home_context'].values
    pool_a_pp = events_bank.loc[mask_a_pp_sample, 'xg_away_context'].values
    
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
        
        # 2. Draw Probabilities from restricted matchup context pools
        xg_h_5v5 = np.random.choice(pool_h_5v5, n_h_5v5, replace=True) if n_h_5v5 > 0 else np.array([])
        xg_a_5v5 = np.random.choice(pool_a_5v5, n_a_5v5, replace=True) if n_a_5v5 > 0 else np.array([])
        xg_h_pp = np.random.choice(pool_h_pp, n_h_pp, replace=True) if n_h_pp > 0 else np.array([])
        xg_a_pp = np.random.choice(pool_a_pp, n_a_pp, replace=True) if n_a_pp > 0 else np.array([])
        
        # 3. Simulate Goals
        h_5v5_goals = np.sum(np.random.rand(n_h_5v5) < xg_h_5v5)
        a_5v5_goals = np.sum(np.random.rand(n_a_5v5) < xg_a_5v5)
        h_pp_goals = np.sum(np.random.rand(n_h_pp) < xg_h_pp)
        a_pp_goals = np.sum(np.random.rand(n_a_pp) < xg_a_pp)
        
        h_total = h_5v5_goals + h_pp_goals
        a_total = a_5v5_goals + a_pp_goals
        
        sim_results['h_goals'].append(h_total)
        sim_results['a_goals'].append(a_total)
        sim_results['h_xg'].append(xg_h_5v5.sum() + xg_h_pp.sum())
        sim_results['a_xg'].append(xg_a_5v5.sum() + xg_a_pp.sum())
        
        # For breakdown (optional)
        if 'h_xg_5v5' not in sim_results:
            sim_results['h_xg_5v5'] = []
            sim_results['a_xg_5v5'] = []
            sim_results['h_xg_pp'] = []
            sim_results['a_xg_pp'] = []
        sim_results['h_xg_5v5'].append(xg_h_5v5.sum())
        sim_results['a_xg_5v5'].append(xg_a_5v5.sum())
        sim_results['h_xg_pp'].append(xg_h_pp.sum())
        sim_results['a_xg_pp'].append(xg_a_pp.sum())
        
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

    # --- DASHBOARD PLOTTING ---
    out_dir = Path(f"analysis/matchups")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plot_matchup_dashboard(
        home, away, 
        sim_results, density_stats,
        dens_h_5v5, dens_a_5v5, dens_h_pp, dens_a_pp,
        out_dir / f"matchup_dashboard_{home}_{away}.png"
    )

def plot_matchup_dashboard(home, away, results, density_stats, dens_h_5v5, dens_a_5v5, dens_h_pp, dens_a_pp, out_path):
    """Generate a single-page summary dashboard for the matchup (Relative to League)."""
    import matplotlib.gridspec as gridspec
    from scipy.ndimage import gaussian_filter
    from puck.rink import draw_rink, rink_half_height_at_x
    import matplotlib.ticker as ticker
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # Process Results
    h_goals = np.array(results['h_goals'])
    a_goals = np.array(results['a_goals'])
    h_xg = np.array(results['h_xg']) 
    a_xg = np.array(results['a_xg'])
    
    mean_h = np.mean(h_goals)
    mean_a = np.mean(a_goals)
    
    # Win Prob (Include Ties split 50/50)
    wins_h = np.sum(h_goals > a_goals)
    wins_a = np.sum(a_goals > h_goals)
    ties = np.sum(h_goals == a_goals) 
    
    prob_h = (wins_h + ties * 0.5) / len(h_goals) * 100
    prob_a = (wins_a + ties * 0.5) / len(a_goals) * 100
    
    # Setup Figure (4 Rows now)
    fig = plt.figure(figsize=(16, 17))
    gs = gridspec.GridSpec(4, 2, height_ratios=[0.15, 0.3, 0.3, 0.25])
    fig.suptitle(f"Matchup Preview: {home} vs {away}", fontsize=24, fontweight='bold', y=0.98)
    
    # 1. SCORE HEADER (Top Row Spanning)
    ax_score = fig.add_subplot(gs[0, :])
    ax_score.axis('off')
    
    # Calculate Dashboard Stats
    h_sec = density_stats['h_matchup_sec']
    a_sec = density_stats['a_matchup_sec']
    h_xg_raw = density_stats['h_matchup_xg']
    a_xg_raw = density_stats['a_matchup_xg']
    
    h_comp_rate = (h_xg_raw / h_sec) * 3600 if h_sec > 0 else 0
    a_comp_rate = (a_xg_raw / a_sec) * 3600 if a_sec > 0 else 0
    
    h_pp_sec = density_stats['h_pp_matchup_sec']
    a_pp_sec = density_stats['a_pp_matchup_sec']
    h_pp_xg_raw = density_stats['h_pp_matchup_xg']
    a_pp_xg_raw = density_stats['a_pp_matchup_xg']
    
    h_pp_comp_rate = (h_pp_xg_raw / h_pp_sec) * 3600 if h_pp_sec > 0 else 0
    a_pp_comp_rate = (a_pp_xg_raw / a_pp_sec) * 3600 if a_pp_sec > 0 else 0
    
    mean_h_xg = np.mean(h_xg)
    mean_a_xg = np.mean(a_xg)
    mean_h_5 = np.mean(results.get('h_xg_5v5', [0]))
    mean_h_p = np.mean(results.get('h_xg_pp', [0]))
    mean_a_5 = np.mean(results.get('a_xg_5v5', [0]))
    mean_a_p = np.mean(results.get('a_xg_pp', [0]))
    
    spread = mean_a - mean_h
    favorite = home if spread < 0 else away
    spread_val = abs(spread)
    
    # CENTER: Score and Spread
    ax_score.text(0.5, 0.7, f"Projected Score\n{home} {mean_h:.1f} - {mean_a:.1f} {away}", ha='center', va='center', fontsize=20, weight='bold')
    ax_score.text(0.5, 0.3, f"Spread: {favorite} -{spread_val:.1f}   |   Total: {mean_h+mean_a:.1f}", ha='center', va='center', fontsize=14, color='darkred')
    
    # LEFT: Home Stats
    ax_score.text(0.2, 0.8, f"{home} Win: {prob_h:.1f}%", ha='center', va='center', fontsize=16, color='blue', weight='bold')
    ax_score.text(0.2, 0.5, f"Projected xG: {mean_h_xg:.2f}\n(5v5: {mean_h_5:.2f}  |  PP: {mean_h_p:.2f})", ha='center', va='center', fontsize=12)
    ax_score.text(0.2, 0.2, f"Base Rates (xtG/60) -> 5v5: {h_comp_rate:.1f} | PP: {h_pp_comp_rate:.1f}", ha='center', va='center', fontsize=11, color='dimgray')
    
    # RIGHT: Away Stats
    ax_score.text(0.8, 0.8, f"{away} Win: {prob_a:.1f}%", ha='center', va='center', fontsize=16, color='orange', weight='bold')
    ax_score.text(0.8, 0.5, f"Projected xG: {mean_a_xg:.2f}\n(5v5: {mean_a_5:.2f}  |  PP: {mean_a_p:.2f})", ha='center', va='center', fontsize=12)
    ax_score.text(0.8, 0.2, f"Base Rates (xtG/60) -> 5v5: {a_comp_rate:.1f} | PP: {a_pp_comp_rate:.1f}", ha='center', va='center', fontsize=11, color='dimgray')
    
    # --- HEATMAP HELPER ---
    extent = [-100, 100, -42.5, 42.5]
    
    def render_heatmap_row(ax, h_grid, a_grid, title, vmax=0.02):
        draw_rink(ax, show_goals=True)
        ax.set_title(title, fontsize=14)
        
        sigma = 6.0
        smooth_h = gaussian_filter(np.fliplr(h_grid), sigma=sigma)
        smooth_a = gaussian_filter(a_grid, sigma=sigma)
        
        rows, cols = smooth_h.shape
        xs = np.linspace(-100, 100, cols)
        ys = np.linspace(-42.5, 42.5, rows)
        X, Y = np.meshgrid(xs, ys)
        
        v_rink_half_height = np.vectorize(rink_half_height_at_x)
        max_y = v_rink_half_height(X)
        mask_rink = np.abs(Y) > max_y
        mask_neutral = np.abs(X) < 25
        full_mask = mask_rink | mask_neutral
        
        final_data = np.full_like(smooth_h, np.nan)
        final_data[:, :100] = smooth_h[:, :100]
        final_data[:, 100:] = smooth_a[:, 100:]
        
        final_masked = np.ma.masked_where(full_mask, final_data)
        
        im = ax.imshow(final_masked, extent=extent, origin='lower', cmap='coolwarm', vmin=-vmax, vmax=vmax, zorder=1)
        
        ax.set_facecolor('white')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        
        cbar = fig.colorbar(im, cax=cax)
        tick_vals = [-vmax, -vmax/2, 0, vmax/2, vmax]
        cbar.locator = ticker.FixedLocator(tick_vals)
        cbar.update_ticks()
        cbar.set_label("Excess xG/60 (per 100 sq ft)", rotation=270, labelpad=15)
        
        ax.axis('off')
        ax.set_frame_on(False)
        
        ax.text(-60, 0, f"{home}\nOffense", ha='center', va='center', fontsize=20, alpha=0.3, fontweight='bold', color='blue')
        ax.text(60, 0, f"{away}\nOffense", ha='center', va='center', fontsize=20, alpha=0.3, fontweight='bold', color='orange')

    # 2. RELATIVE HEATMAP (5v5)
    ax_rink_5v5 = fig.add_subplot(gs[1, :])
    render_heatmap_row(ax_rink_5v5, dens_h_5v5, dens_a_5v5, "5v5 Matchup Density vs League Average\n(Left: Home Expected Goals | Right: Away Expected Goals)", vmax=0.02)
    
    # 3. RELATIVE HEATMAP (PP)
    ax_rink_pp = fig.add_subplot(gs[2, :])
    # The user typically expects the Power Play maps to pop more heavily. 0.05 or 0.08 is a standard PP vmax. 
    # Let's keep it to 0.04 to give it some room since PP densities are much higher than 5v5 densities per 60.
    render_heatmap_row(ax_rink_pp, dens_h_pp, dens_a_pp, "Power Play Matchup Density vs League Average\n(Left: Home Offense | Right: Away Offense)", vmax=0.04)
    
    fig.patch.set_facecolor('white')
    
    # 4. SCORE DISTRIBUTION
    ax_dist = fig.add_subplot(gs[3, :])
    
    bins = np.linspace(0, max(h_xg.max(), a_xg.max()) + 1, 50)
    ax_dist.hist(h_xg, bins=bins, alpha=0.6, label=f"{home} Total xG", color='blue', density=True)
    ax_dist.hist(a_xg, bins=bins, alpha=0.6, label=f"{away} Total xG", color='orange', density=True)
    
    ax_dist.set_title(f"Simulated Matches xG Distribution", fontsize=14)
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)
    
    # Save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved dashboard to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("home", help="Home Team Abbr (e.g. NYR)")
    parser.add_argument("away", help="Away Team Abbr (e.g. NJD)")
    args = parser.parse_args()
    
    simulate_matchup(args.home, args.away, season="20252026")

if __name__ == "__main__":
    main()
