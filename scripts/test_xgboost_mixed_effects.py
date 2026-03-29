
import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import mixed_effects, analyze, data_pipeline
from scripts.evaluate_predictive_power import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_xgboost_mixed_effects():
    # 1. Load Data (Small sample of 2024-25)
    season = "20242025"
    csv_path = analyze.locate_season_csv(season)
    df = pd.read_csv(csv_path)
    
    # Preprocess
    df = data_pipeline.preprocess_features(
        df, is_training=False, apply_imputation=True, 
        apply_arena_adjustments=True, apply_filtering=True,
        impute_alpha=0.2
    )
    
    # 2. Get Model via Registry
    registry = ModelRegistry()
    
    # This should now load XGBoost and fit mixed effects
    logger.info("Instantiating Mixed Effects XGBoost model...")
    model = registry.get_model("mixed_effects_xgboost_nested", train_df=df)
    
    if model is None:
        logger.error("Failed to load model.")
        return
        
    # 3. Verify types
    logger.info(f"Model Type: {type(model)}")
    logger.info(f"Base Model Type: {type(model.base_model_)}")
    
    # 4. Predict
    logger.info("Running predict_proba...")
    probs = model.predict_proba(df)
    
    logger.info(f"Probs shape: {probs.shape}")
    logger.info(f"Mean Prob: {probs[:, 1].mean():.4f}")
    
    # 5. Check coefficients
    coefs = model.get_all_coefficients()
    logger.info(f"Number of learned coefficients: {len(coefs)}")
    if not coefs.empty:
        logger.info("\nTop 5 Offensive Intercepts (5v5):")
        top_off = coefs[(coefs['game_state'] == '5v5') & (coefs['role'] == 'Offense')].sort_values('coef', ascending=False).head(5)
        print(top_off)

    logger.info("Verification Success!")

if __name__ == "__main__":
    test_xgboost_mixed_effects()
