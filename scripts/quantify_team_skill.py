"""scripts/quantify_team_skill.py

Uses Poisson Regression (L2 Regularized) to isolate team-level skill proxies.
Formula: ln(Goals) = 1.0 * ln(xG) + Intercept + Team_Effect
Proxy Multiplier = exp(Intercept + Team_Effect)

Separated by Game State:
- 5v5: Standard Even Strength
- 5v4: Power Play (Offense) / Penalty Kill (Defense)
- 4v5: Shorthanded (Offense) / Power Play (Defense)

Implemented using scipy.optimize to avoid extra dependencies (statsmodels).
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import logging
from pathlib import Path
from scipy.optimize import minimize

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_xgs, analyze, config as puck_config

def get_team_mapping():
    """Fetches team mapping from NHL API or fallback."""
    try:
        resp = requests.get('https://api.nhle.com/stats/rest/en/team')
        resp.raise_for_status()
        teams_data = resp.json().get('data', [])
        return {str(t['id']): t['triCode'] for t in teams_data if 'triCode' in t}
    except Exception as e:
        logger.warning(f"Could not fetch team mapping: {e}")
        return {}

def poisson_loss(params, X, y, log_offset, alpha):
    """
    Negative Log Likelihood for Poisson + L2 Reg.
    params: [intercept, beta_0, beta_1, ...]
    """
    intercept = params[0]
    betas = params[1:]
    
    # log_mu = offset + intercept + X @ betas
    log_mu = log_offset + intercept + X @ betas
    mu = np.exp(log_mu)
    
    # Poisson NLL = sum(mu - y*log_mu)
    # (ignoring log(y!) constant)
    nll = np.sum(mu - y * log_mu)
    
    # L2 Regularization on BETAS only (not intercept)
    reg = alpha * np.sum(betas ** 2)
    
    return nll + reg

def poisson_grad(params, X, y, log_offset, alpha):
    """Gradient of Loss."""
    intercept = params[0]
    betas = params[1:]
    
    log_mu = log_offset + intercept + X @ betas
    mu = np.exp(log_mu)
    
    # dNLL/dtheta = sum((mu - y) * d_log_mu/dtheta)
    diff = mu - y
    
    # Intercept Grad
    d_int = np.sum(diff)
    
    # Beta Grad
    # X is (N, K), diff is (N,)
    d_betas = X.T @ diff
    
    # Add Regularization Grad: 2 * alpha * beta
    d_betas += 2 * alpha * betas
    
    return np.concatenate(([d_int], d_betas))

def fit_skill_model(df, target_col, team_col, xg_col, alpha_val=10.0, label="Global"):
    """
    Fits a Poisson Regression with log(xG) offset using scipy.optimize.
    alpha_val: Regularization strength (L2 penalty).
    """
    # Prepare Data
    df = df.copy().dropna(subset=[target_col, team_col, xg_col])
    
    if len(df) < 50:
        logger.warning(f"Not enough data for {label} (n={len(df)}). Returning empty.")
        return pd.DataFrame()

    # ln(xG) offset
    df['log_xgs'] = np.log(df[xg_col].clip(1e-6))
    log_offset = df['log_xgs'].values
    
    # One-Hot Encode Teams
    team_dummies = pd.get_dummies(df[team_col], prefix='team').astype(float)
    team_names = team_dummies.columns.tolist()
    X = team_dummies.values
    y = df[target_col].values.astype(float)
    
    n_features = X.shape[1]
    
    # Initial Guess
    avg_ratio = np.sum(y) / np.sum(np.exp(log_offset))
    init_intercept = np.log(avg_ratio) if avg_ratio > 0 else 0.0
    initial_params = np.zeros(n_features + 1)
    initial_params[0] = init_intercept
    
    logger.info(f"Fitting {label} Poisson (alpha={alpha_val}) on {len(X)} events...")
    
    res = minimize(
        fun=poisson_loss,
        x0=initial_params,
        args=(X, y, log_offset, alpha_val),
        jac=poisson_grad,
        method='L-BFGS-B',
        options={'maxiter': 2000, 'disp': False}
    )
    
    intercept = res.x[0]
    betas = res.x[1:]
    
    logger.info(f"Fit {label}: Intercept={intercept:.4f} (exp={np.exp(intercept):.4f})")
    
    # Extract Team Effects
    res_list = []
    for i, col in enumerate(team_names):
        tid = col.replace('team_', '')
        beta = betas[i]
        res_list.append({
            'TeamID': tid,
            'Beta': beta,
            'Multiplier': np.exp(intercept + beta)
        })
        
    return pd.DataFrame(res_list)

def main():
    season = "20252026"
    logger.info(f"--- Quantifying Team Skill Proxies for {season} (By State) ---")
    
    # 1. Load Data
    data_path = Path(f"data/{season}/{season}_df.csv")
    if not data_path.exists():
        data_path = Path(f"data/{season}.csv")
    
    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        return

    logger.info(f"Loading data from {data_path}...")
    df = fit_xgs.load_data(str(data_path))
    logger.info(f"Loaded DF Columns: {list(df.columns)}")
    if 'game_state' in df.columns:
        logger.info(f"Unique Game States: {df['game_state'].unique()}")
    
    # 2. Predict xG
    model_path = Path('analysis/xgs/xg_model_nested_all.joblib')
    if not model_path.exists():
        model_path = Path('analysis/xgs/xg_model_nested.joblib')
        
    if not model_path.exists():
        logger.error("No xG model found. Please train it first.")
        return

    logger.info(f"Generating xG predictions using {model_path}...")
    df_pred, _, _ = analyze._predict_xgs(df, model_path=str(model_path))
    
    # Filter to shots
    shot_events = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
    df_pred = df_pred[df_pred['event'].isin(shot_events)].copy()
    
    if 'is_net_empty' in df_pred.columns:
        df_pred = df_pred[df_pred['is_net_empty'] != 1].copy()
    
    if 'is_goal' not in df_pred.columns:
        df_pred['is_goal'] = (df_pred['event'] == 'goal').astype(int)
        
    logger.info(f"Total Goals: {df_pred['is_goal'].sum()}, Total xG: {df_pred['xgs'].sum():.1f}")
    
    # --- PREPARE COLUMNS ---
    team_col = 'team_id' if 'team_id' in df_pred.columns else 'event_team_id'
    
    # Add Shooter/Defender Columns
    if 'home_id' in df_pred.columns and 'away_id' in df_pred.columns:
        # Offense: Blocked shots correspond to the OPPOSING team's shot
        df_pred['shooting_team_id'] = df_pred[team_col]
        mask_blocked = df_pred['event'] == 'blocked-shot'
        mask_home_block = mask_blocked & (df_pred[team_col] == df_pred['home_id'])
        mask_away_block = mask_blocked & (df_pred[team_col] == df_pred['away_id'])
        df_pred.loc[mask_home_block, 'shooting_team_id'] = df_pred.loc[mask_home_block, 'away_id']
        df_pred.loc[mask_away_block, 'shooting_team_id'] = df_pred.loc[mask_away_block, 'home_id']
        
        # Defense: Blocked shots correspond to the ACTUAL team
        df_pred['defending_team_id'] = np.nan
        df_pred.loc[mask_blocked, 'defending_team_id'] = df_pred.loc[mask_blocked, team_col]
        
        mask_offense = ~mask_blocked
        mask_home_off = mask_offense & (df_pred[team_col] == df_pred['home_id'])
        df_pred.loc[mask_home_off, 'defending_team_id'] = df_pred.loc[mask_home_off, 'away_id']
        mask_away_off = mask_offense & (df_pred[team_col] == df_pred['away_id'])
        df_pred.loc[mask_away_off, 'defending_team_id'] = df_pred.loc[mask_away_off, 'home_id']
    else:
        logger.error("Missing home_id/away_id for attribution.")
        return

    # Infer Skater Counts per side if missing
    if ('home_skaters' not in df_pred.columns or 'away_skaters' not in df_pred.columns) and 'game_state' in df_pred.columns:
        logger.info("Parsing skater counts from game_state...")
        # game_state format: "5v5" -> Home v Away
        def parse_state(s):
            if not isinstance(s, str) or 'v' not in s: 
                return 5, 5
            try:
                parts = s.split('v')
                return int(parts[0]), int(parts[1])
            except:
                return 5, 5
                
        parsed = df_pred['game_state'].apply(parse_state)
        df_pred['home_skaters'] = parsed.apply(lambda x: x[0])
        df_pred['away_skaters'] = parsed.apply(lambda x: x[1])
    elif 'home_skaters' not in df_pred.columns:
        logger.warning("Skater counts missing and no game_state! Defaulting to 5v5.")
        df_pred['home_skaters'] = 5
        df_pred['away_skaters'] = 5
    else:
        # Fill NaNs with 5
        df_pred['home_skaters'] = df_pred['home_skaters'].fillna(5)
        df_pred['away_skaters'] = df_pred['away_skaters'].fillna(5)

    # Determine Shooter/Defender Strength based on attribution
    # If shooter is home, strength is Home v Away
    mask_shooter_home = (df_pred['shooting_team_id'] == df_pred['home_id'])
    df_pred['shooter_strength'] = np.where(mask_shooter_home, df_pred['home_skaters'], df_pred['away_skaters'])
    df_pred['defender_strength'] = np.where(mask_shooter_home, df_pred['away_skaters'], df_pred['home_skaters'])
    
    # Define States for Analysis
    states = {
        '5v5': {'shooter': 5, 'defender': 5, 'alpha': 5.0},
        '5v4': {'shooter': 5, 'defender': 4, 'alpha': 10.0}, # PP Offense / PK Defense
        '4v5': {'shooter': 4, 'defender': 5, 'alpha': 20.0}, # SH Offense / PP Defense
    }
    
    proxy_results = {}
    team_map = get_team_mapping()
    
    # Initialize Master DF with just TeamID
    all_team_ids = set(df_pred['shooting_team_id'].unique()) | set(df_pred['defending_team_id'].unique())
    master_df = pd.DataFrame({'TeamID': [str(int(t)) for t in all_team_ids if pd.notna(t)]})
    
    for state_name, config in states.items():
        logger.info(f"Processing State: {state_name} (S:{config['shooter']} v D:{config['defender']})")
        
        # Filter Data
        mask_state = (df_pred['shooter_strength'] == config['shooter']) & \
                     (df_pred['defender_strength'] == config['defender'])
        df_state = df_pred[mask_state].copy()
        
        if len(df_state) < 100:
            logger.warning(f"Skipping {state_name}: Insufficient data ({len(df_state)})")
            continue
            
        # Fit Offense
        off_res = fit_skill_model(df_state, 'is_goal', 'shooting_team_id', 'xgs', 
                                 alpha_val=config['alpha'], label=f"Offense_{state_name}")
        if not off_res.empty:
            off_res = off_res.rename(columns={'Multiplier': f'Offense_{state_name}', 'Beta': f'Offense_Beta_{state_name}'})
            off_res['TeamID'] = off_res['TeamID'].astype(str).str.replace(r'\.0$', '', regex=True)
            master_df = master_df.merge(off_res[['TeamID', f'Offense_{state_name}']], on='TeamID', how='left')

        # Fit Defense
        def_res = fit_skill_model(df_state, 'is_goal', 'defending_team_id', 'xgs',
                                 alpha_val=config['alpha'], label=f"Defense_{state_name}")
        if not def_res.empty:
            def_res = def_res.rename(columns={'Multiplier': f'Defense_{state_name}', 'Beta': f'Defense_Beta_{state_name}'})
            def_res['TeamID'] = def_res['TeamID'].astype(str).str.replace(r'\.0$', '', regex=True)
            master_df = master_df.merge(def_res[['TeamID', f'Defense_{state_name}']], on='TeamID', how='left')

    # Add Team Abbrev
    master_df['TeamAbbrev'] = master_df['TeamID'].map(team_map).fillna(master_df['TeamID'])
    
    # Fill NaN Multipliers with 1.0 (Neutral)
    mult_cols = [c for c in master_df.columns if 'Offense' in c or 'Defense' in c]
    master_df[mult_cols] = master_df[mult_cols].fillna(1.0)
    
    # Reorder
    # Columns: TeamID, TeamAbbrev, Offense_5v5, Defense_5v5, Offense_5v4, Defense_5v4...
    cols_ordered = ['TeamID', 'TeamAbbrev'] + sorted(mult_cols)
    master_df = master_df[cols_ordered]
    
    # Sort by 5v5 Offense
    if 'Offense_5v5' in master_df.columns:
        master_df = master_df.sort_values('Offense_5v5', ascending=False)
    
    # Save
    out_path = Path('analysis/team_skill_proxies.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(out_path, index=False)
    logger.info(f"Skill proxies saved to {out_path}")
    
    print("\n=== TEAM SKILL PROXIES (SPLIT BY STATE) ===")
    print(master_df.head(10))
    
    # Visualize 5v5 vs 5v4 Offense
    if 'Offense_5v5' in master_df.columns and 'Offense_5v4' in master_df.columns:
        plt.figure(figsize=(10, 8))
        plt.scatter(master_df['Offense_5v5'], master_df['Offense_5v4'], alpha=0.7)
        
        # Labels
        for i, row in master_df.iterrows():
            plt.text(row['Offense_5v5'], row['Offense_5v4'], row['TeamAbbrev'], fontsize=8)
            
        plt.xlabel('5v5 Offense Multiplier')
        plt.ylabel('5v4 Offense Multiplier (PP)')
        plt.title('Offensive Skill: 5v5 vs Power Play')
        plt.grid(True, alpha=0.3)
        plt.plot([min(plt.xlim()), max(plt.xlim())], [min(plt.xlim()), max(plt.xlim())], '--', color='gray')
        
        viz_path = Path('analysis/skill_proxy_state_compare.png')
        plt.savefig(viz_path)
        logger.info(f"State comparison viz saved to {viz_path}")

if __name__ == "__main__":
    main()
