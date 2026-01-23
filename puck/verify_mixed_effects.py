"""verify_mixed_effects.py

Verification script for Mixed Effects xG Model.
Focuses on Team-Effects to ensure robust plotting and validation.
Using 2025-2026 data only.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from puck import mixed_effects, fit_nested_xgs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyME")

def verify_team_effects():
    # 1. Load Data (Specific to 20252026)
    logger.info("Loading 2025-2026 data...")
    try:
        # Direct Load to avoid glob issues
        # Try file first
        p_file = Path("data/20252026.csv")
        p_dir = Path("data/20252026")
        
        if p_file.exists():
            logger.info(f"Loading direct file: {p_file}")
            df = pd.read_csv(p_file)
        elif p_dir.exists():
            logger.info(f"Loading from directory: {p_dir}")
            files = list(p_dir.glob("*.csv"))
            if not files:
                raise FileNotFoundError(f"No CSVs in {p_dir}")
            df = pd.concat([pd.read_csv(f) for f in files])
        else:
            raise FileNotFoundError("Could not find 20252026 data file or folder.")
            
        logger.info(f"Raw rows: {len(df)}")
        
        # Preprocessing
        df = fit_nested_xgs.preprocess_features(df)
        logger.info(f"Preprocessed rows: {len(df)}")
        
        # 2. Enrich with Team Name
        # Logic: If team_id == home_id -> home_abb, else away_abb
        logger.info("Enriching Team Names...")
        
        def get_team_name(row):
            # Ensure types match
            tid = row['team_id']
            hid = row['home_id']
            aid = row['away_id']
            # Parse if strings/ints mixed
            try:
                if float(tid) == float(hid): return row['home_abb']
                if float(tid) == float(aid): return row['away_abb']
            except:
                pass
            return "UNKNOWN"

        if 'team_name' not in df.columns:
            if 'home_abb' in df.columns and 'away_abb' in df.columns:
                df['team_name'] = df.apply(get_team_name, axis=1)
                
                # Check how many unknowns
                n_unknown = (df['team_name'] == 'UNKNOWN').sum()
                if n_unknown > 0:
                    logger.warning(f"Found {n_unknown} rows with UNKNOWN team name.")
            else:
                # Fallback to team_id if abbs missing
                logger.warning("home_abb/away_abb missing, using team_id.")
                df['team_name'] = df['team_id'].astype(str)
                
        # Filter for valid team names
        df_clean = df[df['team_name'] != "UNKNOWN"].copy()
        if len(df_clean) == 0:
             logger.warning("All team names UNKNOWN? Reverting to all data but using team_id/home/away logic fix.")
             # Fallback: Just use team_id as str
             df['team_name'] = df['team_id'].astype(str)
             df_clean = df
             
        df = df_clean
        logger.info(f"Final rows for train/test: {len(df)}")
        
        # Enrich features (Handedness etc)
        from puck import fit_xgs
        df = fit_xgs.enrich_data_with_bios(df)
        
    except Exception as e:
        logger.error(f"Failed to load/preprocess data: {e}")
        return

    if len(df) < 50:
        logger.error("Not enough data to split/train.")
        return

    # 2. Split Train/Test
    # We want to fit on Train, Evaluate on Test to see if random slopes generalize
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    logger.info(f"Train/Test Split: {len(train_df)} / {len(test_df)}")

    # 3. Fit Mixed Effects Model (Team Level)
    logger.info("Fitting Mixed Effects Model (Group: team_name)...")
    me_model = mixed_effects.MixedEffectsXG(
        n_estimators=100, 
        l2_reg=1.0,
        learning_rate=0.5,
        group_col='team_name'
    )
    me_model.fit(train_df)

    # 4. Evaluate Performance
    logger.info("Evaluating...")
    
    # Base Model Logic (Manual reconstruction for comparison)
    p_base = me_model.base_model_.predict_proba(test_df)[:, 1]
    p_me = me_model.predict_proba(test_df)[:, 1]
    
    y_test = (test_df['event'] == 'goal').astype(int)
    
    ll_base = log_loss(y_test, p_base)
    ll_me = log_loss(y_test, p_me)
    
    logger.info(f"Log Loss Comparison (Test Set):")
    logger.info(f"  Base Model: {ll_base:.5f}")
    logger.info(f"  Mixed Eff : {ll_me:.5f}")
    logger.info(f"  Improvement: {ll_base - ll_me:.6f}")
    
    # 5. Robust Plotting: Team Performance
    visuals_dir = Path("analysis/mixed_effects_verification_2026")
    visuals_dir.mkdir(parents=True, exist_ok=True)
    
    test_df['p_base'] = p_base
    test_df['p_me'] = p_me
    test_df['is_goal'] = y_test
    
    # --- Plot A: Log Loss Improvement per Team ---
    team_metrics = []
    for team, grp in test_df.groupby('team_name'):
        if len(grp) < 10: continue 
        
        try:
            ll_b = log_loss(grp['is_goal'], grp['p_base'], labels=[0,1])
            ll_m = log_loss(grp['is_goal'], grp['p_me'], labels=[0,1])
            
            team_metrics.append({
                'Team': team,
                'Base LogLoss': ll_b,
                'Mixed LogLoss': ll_m,
                'Improvement': ll_b - ll_m,
                'Goals': grp['is_goal'].sum(),
                'Shots': len(grp)
            })
        except Exception:
            pass
        
    if team_metrics:
        metrics_df = pd.DataFrame(team_metrics).sort_values('Improvement', ascending=False)
        
        plt.figure(figsize=(14, 8))
        sns.barplot(data=metrics_df, x='Improvement', y='Team', palette='coolwarm')
        plt.axvline(0, color='black', linestyle='-', linewidth=1)
        plt.title(f'Log Loss Improvement by Team (2025-2026) - Positive is Better\nOverall Improv: {ll_base - ll_me:.5f}')
        plt.xlabel('Log Loss Reduction (Base - Mixed)')
        plt.tight_layout()
        plt.savefig(visuals_dir / 'team_logloss_improvement_2026.png')
        plt.close()
        
        # --- Plot B: Calibration / Totals ---
        # Group by Team and Calculate Sums
        team_sums = test_df.groupby('team_name')[['p_base', 'p_me', 'is_goal']].sum().reset_index()
        team_sums['Base Diff'] = team_sums['p_base'] - team_sums['is_goal']
        team_sums['Mixed Diff'] = team_sums['p_me'] - team_sums['is_goal']
        
        # Sort by Base Error magnitude
        team_sums = team_sums.sort_values('Base Diff', key=abs, ascending=False)
        
        plt.figure(figsize=(14, 8))
        x = np.arange(len(team_sums))
        width = 0.35
        
        plt.bar(x - width/2, team_sums['Base Diff'], width, label='Base Model Error', alpha=0.7)
        plt.bar(x + width/2, team_sums['Mixed Diff'], width, label='Mixed Model Error', alpha=0.7)
        
        plt.xticks(x, team_sums['team_name'], rotation=90)
        plt.axhline(0, color='black')
        plt.ylabel('Predicted Goals - Actual Goals')
        plt.title('Prediction Error by Team (Test Set) - Closer to 0 is Better')
        plt.legend()
        plt.tight_layout()
        plt.savefig(visuals_dir / 'team_goals_error_2026.png')
        plt.close()

    logger.info(f"Verification plots saved to {visuals_dir}")

if __name__ == "__main__":
    verify_team_effects()
