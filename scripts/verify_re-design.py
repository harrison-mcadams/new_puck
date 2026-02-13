"""verify_re-design.py

Verification script for the new Split Offense/Defense Mixed Effects Model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import joblib
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import mixed_effects, fit_nested_xgs, fit_xgs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyRedesign")

def verify_redesign():
    # 1. Load Data (2025-2026)
    logger.info("Loading 2025-2026 data...")
    try:
        # Try single file first (preferred)
        p_file = Path("data/20252026.csv")
        if p_file.exists():
            logger.info(f"Loading {p_file}...")
            df = pd.read_csv(p_file)
        else:
            p_dir = Path("data/20252026")
            if p_dir.exists():
                files = list(p_dir.glob("*.csv"))
                if not files:
                    raise FileNotFoundError(f"No CSVs in {p_dir}")
                df = pd.concat([pd.read_csv(f) for f in files])
            else:
                logger.error("Could not find data.")
                return
                
        logger.info(f"Raw rows: {len(df)}")
        
        # Preprocessing
        df = fit_nested_xgs.preprocess_features(df)
        logger.info(f"Preprocessed rows: {len(df)}")
        
        # Enrich
        df = fit_xgs.enrich_data_with_bios(df)
        
        # Ensure Team Names (Robust ID -> Abb)
        if 'team_abbrev' in df.columns:
            df['off_team_name'] = df['team_abbrev']
        elif 'home_id' in df.columns and 'home_abb' in df.columns:
             # Map using ID match
             is_home = (df['team_id'] == df['home_id'])
             df['off_team_name'] = np.where(is_home, df['home_abb'], df['away_abb'])
        else:
            df['off_team_name'] = df['team_id'].astype(str)
            
        # Defense Name Logic
        if 'home_abb' in df.columns and 'away_abb' in df.columns:
            # We assume off_team_name is now an abbrev (or ID matching abb if strings)
            # Safe comparison
            is_home_s = (df['off_team_name'] == df['home_abb'])
            df['def_team_name'] = np.where(is_home_s, df['away_abb'], df['home_abb'])
        elif 'home_id' in df.columns:
             # Fallback to ID
             is_home_i = (df['team_id'] == df['home_id'])
             df['def_team_name'] = np.where(is_home_i, df['away_id'], df['home_id']).astype(str)
        else:
            df['def_team_name'] = 'Unknown'
            
        # Filter Unknowns
        df = df[df['def_team_name'] != 'Unknown']
        
    except Exception as e:
        logger.error(f"Data Load Error: {e}")
        return

    # 2. Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # 3. Train New Model
    logger.info("Initializing GameMixedEffectsXG (Split Off/Def)...")
    # We assume base model exists or let it load default
    model = mixed_effects.GameMixedEffectsXG()
    
    # Check if base model file exists, otherwise we might fail
    if not Path(model.base_model_path or "analysis/xgs/xg_model_nested_tensor.joblib").exists():
        logger.warning("Base model not found. Training a dummy base model for testing? Or expect failure.")
        # If real base model missing, we might need to train one quickly or mock it.
        # But usually 'analysis/xgs/...' exists in user env.
        pass

    logger.info("Fitting Model...")
    model.fit(train_df)
    
    # 4. Evaluate
    logger.info("Evaluating on Test Set...")
    probs = model.predict_proba(test_df)[:, 1]
    
    # Base comparison
    if model.base_model_:
        base_probs = model.base_model_.predict_proba(test_df)[:, 1]
        ll_base = log_loss(test_df['event']=='goal', base_probs)
        logger.info(f"Base Model Log Loss: {ll_base:.5f}")
    else:
        ll_base = 0
        
    ll_me = log_loss(test_df['event']=='goal', probs)
    logger.info(f"Mixed  Model Log Loss: {ll_me:.5f}")
    logger.info(f"Improvement: {ll_base - ll_me:.5f}")
    
    # 5. Save Summary & Plots
    out_dir = Path("analysis/verification_redesign")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_summary(str(out_dir))
    
    # 6. Additional Plots (Off vs Def)
    # We can plot the distribution of Offense vs Defense adjustments on the test set
    # Predict margins for components
    test_df['off_adj'] = 0.0
    test_df['def_adj'] = 0.0
    
    # We need to access submodels manually or add a helper?
    # We can use the predict logic partially or just trust the impact analysis in save_summary.
    # Let's use the coefficients from save_summary outputs which generate the scatter plot.
    
    logger.info(f"Verification artifacts saved to {out_dir}")

if __name__ == "__main__":
    verify_redesign()
