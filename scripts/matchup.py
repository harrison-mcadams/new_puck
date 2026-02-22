
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
    
    for state in ['5v5', '5v4', '4v5']:
        mask = events_bank['game_state'] == state
        if not mask.any(): continue
        
        # Subset for this state
        df_state = events_bank[mask].copy()
        
        # 1. Home Offense Context (Home Team vs Away Team)
        df_home = df_state.copy()
        df_home['off_team_name'] = home_team
        df_home['def_team_name'] = away_team
        # Ensure dummy columns for GLM exist (fill 0 if missing in export, shouldn't happen with full export)
        # Predict uses global model now, handles states internally
        probs_home = model.predict_proba(df_home)[:, 1]
        events_bank.loc[mask, 'xg_home_context'] = probs_home
        
        # 2. Away Offense Context (Away Team vs Home Team)
        df_away = df_state.copy()
        df_away['off_team_name'] = away_team
        df_away['def_team_name'] = home_team
        # Predict uses global model now, handles states internally
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

def simulate_matchup(home, away, season, n_sims=1000):
    print(f"\n--- Simulating {home} vs {away} (Monte Carlo N={n_sims}) ---")
    
    # 1. Load Assets
    print("Loading empirical data and models...")
    model, grids, stats, events_bank = load_assets(season)
    
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
    
    # Flip Def Grid (Stored on Right) to match Off Grid (Stored on Left)
    dens_h_5v5 = get_matchup_density(grids[home]['5v5']['grid_for'], np.fliplr(grids[away]['5v5']['grid_against']), BIN_X, BIN_Y)
    
    # Away 5v5 Rates
    a_cf60 = stats[away]['5v5']['attempts_for'] / (stats[away]['5v5']['seconds']/3600) if stats[away]['5v5']['seconds'] else 60.0
    h_ca60 = stats[home]['5v5']['attempts_against'] / (stats[home]['5v5']['seconds']/3600) if stats[home]['5v5']['seconds'] else 60.0
    rate_a_5v5 = np.sqrt(a_cf60 * h_ca60) * (duration_5v5_min/60)
    
    dens_a_5v5 = get_matchup_density(grids[away]['5v5']['grid_for'], np.fliplr(grids[home]['5v5']['grid_against']), BIN_X, BIN_Y)
    
    # 2. Special Teams (Assume 6 mins PP each)
    duration_pp_min = 6.0
    
    # Home PP
    h_pp_cf60 = stats[home]['5v4']['attempts_for'] / (stats[home]['5v4']['seconds']/3600) if stats[home]['5v4']['seconds'] else 60.0
    a_pk_ca60 = stats[away]['4v5']['attempts_against'] / (stats[away]['4v5']['seconds']/3600) if stats[away]['4v5']['seconds'] else 60.0
    rate_h_pp = np.sqrt(h_pp_cf60 * a_pk_ca60) * (duration_pp_min/60)
    
    dens_h_pp = get_matchup_density(grids[home]['5v4']['grid_for'], np.fliplr(grids[away]['4v5']['grid_against']), BIN_X, BIN_Y)
    
    # Away PP
    a_pp_cf60 = stats[away]['5v4']['attempts_for'] / (stats[away]['5v4']['seconds']/3600) if stats[away]['5v4']['seconds'] else 60.0
    h_pk_ca60 = stats[home]['4v5']['attempts_against'] / (stats[home]['4v5']['seconds']/3600) if stats[home]['4v5']['seconds'] else 60.0
    rate_a_pp = np.sqrt(a_pp_cf60 * h_pk_ca60) * (duration_pp_min/60)
    
    dens_a_pp = get_matchup_density(grids[away]['5v4']['grid_for'], np.fliplr(grids[home]['4v5']['grid_against']), BIN_X, BIN_Y)
    
    print(f"  {home} 5v5: {rate_h_5v5:.1f}, PP: {rate_h_pp:.1f}")
    print(f"  {away} 5v5: {rate_a_5v5:.1f}, PP: {rate_a_pp:.1f}")
    
    # --- PRE-CALCULATION ---
    if events_bank is not None:
        events_bank = precalculate_matchup_xg(events_bank, model, home, away)
    
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
        xg_a_p, g_a_p, actual_a_p = sim_batch_precalc(n_a_pp, False, '4v5')  # Away PP = global state '4v5' (home shorthanded)

        
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
    
    print(f"\\n--- Final Results ({n_sims} Games) ---")
    print(f"Win Prob: {home} {100*wins_h/n_sims:.1f}% | {away} {100*wins_a/n_sims:.1f}%")
    print(f"Mean Score: {home} {np.mean(h_goals):.2f} - {away} {np.mean(a_goals):.2f}")
    print(f"Mean xG:    {home} {np.mean(sim_results['h_xg']):.2f} - {away} {np.mean(sim_results['a_xg']):.2f}")
    
    # Show breakdown
    if 'h_xg_5v5' in sim_results:
        print(f"\\nxG Breakdown (Mean):")
        print(f"  {home}: 5v5={np.mean(sim_results['h_xg_5v5']):.2f}, PP={np.mean(sim_results['h_xg_pp']):.2f}")
        print(f"  {away}: 5v5={np.mean(sim_results['a_xg_5v5']):.2f}, PP={np.mean(sim_results['a_xg_pp']):.2f}")
        
        if 'h_actual_5v5' in sim_results:
            print(f"\\n[DEBUG] Actual Goals from Sampled Events (Mean):")
            print(f"  {home}: 5v5={np.mean(sim_results['h_actual_5v5']):.2f}, PP={np.mean(sim_results['h_actual_pp']):.2f}")
            print(f"  {away}: 5v5={np.mean(sim_results['a_actual_5v5']):.2f}, PP={np.mean(sim_results['a_actual_pp']):.2f}")
            print(f"  (These are goals that actually happened in the sampled historical events)")

    # --- LEAGUE BASELINE ---
    print("  Calculating League Baseline Density...")
    l_grid_for = np.zeros_like(grids[home]['5v5']['grid_for'])
    l_grid_ag = np.zeros_like(grids[home]['5v5']['grid_against'])
    n_teams = 0
    for t in grids:
        if '5v5' in grids[t]:
            l_grid_for += grids[t]['5v5']['grid_for']
            l_grid_ag += grids[t]['5v5']['grid_against']
            n_teams += 1
    
    if n_teams > 0:
        l_grid_for /= n_teams
        l_grid_ag /= n_teams
        
    # League Benchmarks (Flip defense just like matchups)
    league_dens_5v5 = get_matchup_density(l_grid_for, np.fliplr(l_grid_ag), BIN_X, BIN_Y)
    
    # Avoid div by zero in relative calc
    league_dens_5v5 = np.maximum(league_dens_5v5, 1e-6)

    # --- DASHBOARD PLOTTING ---
    out_dir = Path(f"analysis/matchups")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plot_matchup_dashboard(
        home, away, 
        sim_results, 
        dens_h_5v5, dens_a_5v5, league_dens_5v5,
        out_dir / f"matchup_dashboard_{home}_{away}.png"
    )

def plot_matchup_dashboard(home, away, results, dens_h, dens_a, league_dens, out_path):
    """Generate a single-page summary dashboard for the matchup (Relative to League)."""
    import matplotlib.gridspec as gridspec
    from scipy.ndimage import gaussian_filter
    from puck.rink import draw_rink, RINK_LENGTH, RINK_WIDTH, rink_half_height_at_x
    
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
    
    # Setup Figure
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.15, 0.45, 0.4])
    fig.suptitle(f"Matchup Preview: {home} vs {away}", fontsize=24, fontweight='bold', y=0.98)
    
    # 1. SCORE HEADER (Top Row Spanning)
    ax_score = fig.add_subplot(gs[0, :])
    ax_score.axis('off')
    
    score_text = (
        f"Projected Score\\n"
        f"{home} {mean_h:.1f} - {mean_a:.1f} {away}"
    )
    ax_score.text(0.5, 0.5, score_text, ha='center', va='center', fontsize=20, weight='bold')
    
    ax_score.text(0.2, 0.5, f"{home} Win\\n{prob_h:.1f}%", ha='center', va='center', fontsize=16, color='blue')
    ax_score.text(0.8, 0.5, f"{away} Win\\n{prob_a:.1f}%", ha='center', va='center', fontsize=16, color='orange')
    
    # 2. RELATIVE HEATMAP (Merged Rink)
    extent = [-100, 100, -42.5, 42.5]
    ax_rink = fig.add_subplot(gs[1, :]) # Span middle row
    draw_rink(ax_rink, show_goals=True)
    
    ax_rink.set_title(f"Matchup Density vs League Average\\n(Left: {home} Offense | Right: {away} Offense)", fontsize=14)
    
    # Compute Relative Density (Ratio). 1.0 = Average.
    # Handle potential zeros in league_dens (already clamped to 1e-6 but good to be safe)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_h = dens_h / league_dens
        rel_a = dens_a / league_dens
    
    # Flip Away to show on Right side (since it was calculated relative to Home Def on Left)
    rel_a_flipped = np.fliplr(rel_a)
    
    # Create Masked Merged Grid
    # Left Side: Home (Cols 0-90 approx, covering -100 to -10)
    # Right Side: Away (Cols 110-200 approx, covering 10 to 100)
    # Center (90-110): Masked
    
    # --- MASKS ---
    rows, cols = rel_h.shape
    xs = np.linspace(-100, 100, cols)
    ys = np.linspace(-42.5, 42.5, rows)
    X, Y = np.meshgrid(xs, ys)
    
    # 1. Rink Boundary Mask
    # Check Y against rink half-height at each X
    v_rink_half_height = np.vectorize(rink_half_height_at_x)
    max_y = v_rink_half_height(X)
    mask_rink = np.abs(Y) > max_y
    
    # 2. Neutral Zone Mask (|X| < 25)
    mask_neutral = np.abs(X) < 25
    
    # Combine Masks
    full_mask = mask_rink | mask_neutral
    
    # --- SMOOTHING ---
    sigma = 1.5
    smooth_h = gaussian_filter(rel_h, sigma=sigma)
    smooth_a = gaussian_filter(rel_a_flipped, sigma=sigma)
    
    # --- MERGE ---
    final_data = np.full_like(smooth_h, np.nan)
    
    # Left Half
    final_data[:, :100] = smooth_h[:, :100]
    # Right Half
    final_data[:, 100:] = smooth_a[:, 100:]
    
    # Apply Mask
    final_masked = np.ma.masked_where(full_mask, final_data)
    
    # Plot using Diverging Colormap (Blue < 1.0 < Red)
    # vmin=0.0 means 0 density (Cold). 1.0 is White. 2.5 is 2.5x Average (Hot).
    im = ax_rink.imshow(final_masked, extent=extent, origin='lower', cmap='coolwarm', vmin=0.0, vmax=2.5, zorder=1)
    
    cbar = plt.colorbar(im, ax=ax_rink, fraction=0.02, pad=0.04)
    cbar.set_label("Relative Density (1.0 = League Avg)", fontsize=10)
    
    # Annotations
    ax_rink.text(-60, 0, f"{home}\\nOffense", ha='center', va='center', fontsize=20, alpha=0.3, fontweight='bold', color='blue')
    ax_rink.text(60, 0, f"{away}\\nOffense", ha='center', va='center', fontsize=20, alpha=0.3, fontweight='bold', color='orange')

    # 3. SCORE DISTRIBUTION (Bottom Row Spanning)
    ax_dist = fig.add_subplot(gs[2, :])
    
    bins = np.linspace(0, max(h_xg.max(), a_xg.max()) + 1, 50)
    ax_dist.hist(h_xg, bins=bins, alpha=0.6, label=f"{home} xG", color='blue', density=True)
    ax_dist.hist(a_xg, bins=bins, alpha=0.6, label=f"{away} xG", color='orange', density=True)
    
    ax_dist.set_title(f"Simulated xG Distribution", fontsize=14)
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
