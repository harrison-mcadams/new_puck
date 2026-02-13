import sys
import os

# Add CWD
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import logging
from puck import data_pipeline

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load Model
    model_path = "analysis/xgs/mixed_effects_v2.joblib"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Load Data (Manual CSV + Pipeline Preprocess)
    data_path = "data/20252026.csv"
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Filter for standard shot events (matches update_mixed_effects.py)
    target_events = ['Goal', 'Shot', 'Missed Shot', 'Blocked Shot', 'goal', 'shot', 'missed-shot', 'blocked-shot', 'shot-on-goal']
    df = df[df['event'].isin(target_events)]
    
    print("Preprocessing features (imputation, bio, arena)...")
    # Need to handle potential missing team abbrevs if pipeline depends on them?
    # pipeline.preprocess_features handles bio enrichment which needs roster lookup.
    # It assumes team_id is present.
    
    df = data_pipeline.preprocess_features(
        df,
        is_training=False,
        apply_imputation=True,
        apply_arena_adjustments=True,
        apply_bio_enrichment=True,
        apply_filtering=True
    )
    
    # 2b. Enrich Team Abbreviations Manually (simplified for PHI)
    # We need 'home_abb' etc for the model's team lookup if it uses unknown team handling
    # But since we just want to debug predictions for PHI, let's just ensure PHI is identified.
    # The MixedEffects model maps team strings to indices.
    
    # Try to use existing cols if pipeline added them
    print(f"Columns after prep: {df.columns.tolist()[:10]}")
    
    # Filter PHI
    # Check what columns we have for teams
    if 'home_abb' in df.columns:
        df_phi = df[(df['home_abb'] == 'PHI') | (df['away_abb'] == 'PHI')].copy()
    elif 'home_team' in df.columns:
        # Assuming team names are full strings or abbrevs?
        # Usually they are names.
        df_phi = df[df['home_team'].str.contains('Phi', case=False) | df['away_team'].str.contains('Phi', case=False)].copy()
        
        # If we don't have abbrevs, the model might fail to look them up if it expects abbrevs.
        # Let's mock them for PHI.
        df_phi['home_team'] = df_phi['home_team'].replace({'Philadelphia Flyers': 'PHI'})
        df_phi['away_team'] = df_phi['away_team'].replace({'Philadelphia Flyers': 'PHI'})
        
        # Also need off_team_name / def_team_name to match PHI
        # But prepare_features handles this logic via config.
    else:
        print("Cannot find team columns!")
        return

    print(f"Stats for PHI subset ({len(df_phi)} rows):")
    
    # Run Predict Steps Manually to Inspect
    # 1. Base
    print("Predicting Base Model...")
    base_probs = model.base_model_.predict_proba(df_phi)[:, 1]
    print(f"Base Probs: Mean={base_probs.mean():.4f}, Sum={base_probs.sum():.2f}")
    
    # 2. Margins
    eps = 1e-6
    base_probs_clip = np.clip(base_probs, eps, 1-eps)
    base_margins = np.log(base_probs_clip / (1 - base_probs_clip))
    print(f"Base Margins: Mean={base_margins.mean():.4f}")
    
    # 3. Features
    print("Preparing Features...")
    # Hack: Access protected method
    X_trans = model._prepare_features(df_phi, fit=False)
    X_trans['game_state'] = df_phi['game_state']
    
    # INSPECT MODEL MAPPING
    if '5v5' in model.models_:
        m5 = model.models_['5v5']
        # team_idx_ is a dict mapping team_name -> int
        # Print first few keys to see what model expects
        if hasattr(m5, 'team_idx_'):
             keys = list(m5.team_idx_.keys())
             print(f"Model 5v5 Team Mapping keys (sample): {keys[:5]}")
             # Check if 'PHI' is in there
             print(f"Is 'PHI' in model map? {'PHI' in m5.team_idx_}")
    
    # Ensure Team Cols exist
    if 'team_abbrev' not in df_phi.columns:
         print("Injecting PHI abbrevs based on team_id=4...")
         df_phi['team_abbrev'] = df_phi['team_id'].apply(lambda x: 'PHI' if x == 4 else 'OPP')
         
    if 'team_abbrev' in df_phi.columns:
         X_trans['off_team_name'] = df_phi['team_abbrev']
    
    if 'home_abb' not in df_phi.columns:
        # data_pipeline might not have added it if we used manual CSV load without enrichment?
        # But we called preprocess_features.
        # Let's fake it if needed for the debug script to run
        df_phi['home_abb'] = df_phi.apply(lambda x: 'PHI' if x.get('home_team') == 'Philadelphia Flyers' else 'OPP', axis=1)
        df_phi['away_abb'] = df_phi.apply(lambda x: 'PHI' if x.get('away_team') == 'Philadelphia Flyers' else 'OPP', axis=1)

    # infer def
    if 'off_team_name' in X_trans.columns:
        # Force valid teams for testing adjustments
        # We know model has 'ANA', 'BOS', etc.
        # We want to see PHI adjustment.
        # Let's set Def Team to 'ANA' (League Average-ish?) or just valid.
        
        print("Forcing Valid Team Names for Feature Mapping...")
        off_vec = pd.Series(['PHI'] * len(df_phi), index=df_phi.index) # Force PHI
        def_vec = pd.Series(['ANA'] * len(df_phi), index=df_phi.index) # Force valid opp
        
        X_trans['off_team_name'] = off_vec
        X_trans['def_team_name'] = def_vec
        
        # Note: True defensive team varies, but for magnitude check this is sufficient.
        # If model expects 'PHI' vs 'ANA', we get that adjustment.
    else:
        # Fallback
        X_trans['off_team_name'] = 'Unknown'
        X_trans['def_team_name'] = 'Unknown'
    
    # 4. State Adj
    final_margins = base_margins.copy()
    
    for state in ['5v5', '5v4', '4v5']:
        mask = df_phi['game_state'] == state
        if not mask.any():
            continue
            
        print(f"--- State {state} ---")
        if state in model.models_:
            sub_model = model.models_[state]
            X_sub = X_trans[mask]
            
            adj = sub_model.predict_margin(X_sub, off_team_col='off_team_name', def_team_col='def_team_name')
            print(f"  Adjustment: Mean={adj.mean():.4f}, Min={adj.min():.4f}, Max={adj.max():.4f}")
            
            # SIMULATE THE BUG?
            # Current code uses ASSIGNMENT
            assigned = adj
            added = base_margins[mask] + adj
            
            print(f"  Result if ASSIGNED: Mean Margin={assigned.mean():.4f} -> Mean Prob={ (1/(1+np.exp(-assigned))).mean():.4f}")
            print(f"  Result if ADDED:    Mean Margin={added.mean():.4f}    -> Mean Prob={ (1/(1+np.exp(-added))).mean():.4f}")
            
            # Apply what the code does?
            # We want to verify what the code DOES.
            # Code: final_margins[mask] = state_adj
            final_margins[mask] = adj # Replicating current implementation
            
    # Final
    final_probs = 1 / (1 + np.exp(-final_margins))
    print(f"\nFinal Predictions (Current Impl): Mean={final_probs.mean():.4f}, Sum={final_probs.sum():.2f}")
    
    actual_goals = (df_phi['event'] == 'goal').sum()
    print(f"Actual Goals: {actual_goals}")
    
    # SUMMARY STATS (AUDIT)
    print("\n--- AUDIT SUMMARY ---")
    print(f"Base Margins (5v5): Mean={base_margins.mean():.4f}, Std={base_margins.std():.4f}")
    if state in model.models_: # Use last state processed (likely '4v5', need to specify)
         pass 

    # Check 5v5 specifically
    if '5v5' in model.models_:
        sub = model.models_['5v5']
        # Extract weights using get_dump
        if hasattr(sub, 'booster_') and sub.booster_ is not None:
             # Legacy
             dump = sub.booster_.get_dump(dump_format='json')
             # ... (keep existing logic if needed but wrap it safely)
             pass
        elif hasattr(sub, 'coef_') and sub.coef_ is not None:
             print(f"  Coefficients loaded. Mean={sub.coef_.mean():.4f}, Std={sub.coef_.std():.4f}")
        else:
             print("  No model found for state.")
        # The original 'except' was orphaned, removing it as it's syntactically incorrect here.
        
        # Check adjustment magnitude
        mask_5v5 = df_phi['game_state'] == '5v5'
        if mask_5v5.any():
            adj_5v5 = sub.predict_margin(X_trans[mask_5v5], off_team_col='off_team_name', def_team_col='def_team_name')
            print(f"5v5 Adjustment: Mean={adj_5v5.mean():.4f}")
            print(f"5v5 Base Margin: Mean={base_margins[mask_5v5].mean():.4f}")
            
            # Key Audit Metric
            print(f"Total Predicted Margin (Base + Adj): Mean={ (base_margins[mask_5v5] + adj_5v5).mean():.4f}")
            print(f"Total Predicted Margin (Adj Only):   Mean={ adj_5v5.mean():.4f}")


if __name__ == "__main__":
    main()
