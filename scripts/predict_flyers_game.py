
import sys
import os
import pandas as pd
import numpy as np
import joblib
import json
import logging
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import fit_xgs, fit_nested_xgs, impute

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("PredictFlyers")

def get_latest_flyers_game(season=20242025):
    """Find the most recent game involving PHI."""
    logger.info(f"Looking for latest Flyers game in {season}...")
    try:
        # Check if we have the season data loaded in data/
        season_path = Path(f'data/{season}/{season}_df.csv')
        if season_path.exists():
            df = pd.read_csv(season_path)
            # Filter for PHI
            phi_games = df[df['team_name'] == 'Philadelphia Flyers'] # Or 'PHI' depending on column
            if phi_games.empty:
                # Try abbreviations if team_name is not full
                phi_games = df[df['team'] == 'PHI']
            
            if not phi_games.empty:
                # Sort by game_id (proxy for time) or date if available
                last_game_id = phi_games['game_id'].max()
                logger.info(f"Found latest game ID in local data: {last_game_id}")
                return last_game_id
                
    except Exception as e:
        logger.warning(f"Could not check local data: {e}")
        
    # Fallback
    return None

def main():
    # 1. Load Models
    logger.info("Loading models...")
    try:
        clf_single = joblib.load('analysis/xgs/xg_model_single.joblib')
        
        # Load Single Model Metadata for Encoding
        try:
            with open('analysis/xgs/xg_model_single.joblib.meta.json', 'r') as f:
                meta_single = json.load(f)
                raw_features_single = ['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type']
                cat_map_single = meta_single.get('categorical_levels_map', {})
        except Exception as e:
            logger.warning(f"Could not load Single Model metadata: {e}")
            raw_features_single = ['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type']
            cat_map_single = {}
        
        # Wrapped check
        if not isinstance(clf_single, fit_xgs.SingleXGClassifier):
             pass 

        clf_nested = joblib.load('analysis/xgs/xg_model_nested.joblib')
        logger.info("Models loaded.")
    except FileNotFoundError:
        logger.error("Models not found! Is training complete?")
        return

    # 2. Get Game Data
    # Assuming backfill just ran for 20252026
    season_path = Path('data/20252026/20252026_df.csv') 
    if not season_path.exists():
         season_path = Path('data/20242025/20242025_df.csv')
    
    if not season_path.exists():
        logger.error("No season data found.")
        return

    logger.info(f"Loading season data from {season_path}...")
    df = pd.read_csv(season_path)
    
    # Filter for Flyers (PHI) using home_abb / away_abb
    logger.info("Searching for Flyers games...")
    
    mask_phi = (df['home_abb'] == 'PHI') | (df['away_abb'] == 'PHI')
    df_phi_games = df[mask_phi]
    
    if df_phi_games.empty:
        logger.error("No Flyers games found in this season file.")
        return
        
    # Get last game
    last_game_id = df_phi_games['game_id'].max()
    logger.info(f"Analyzing most recent Flyers game: {last_game_id}")
    
    df_game = df[df['game_id'] == last_game_id].copy()
    logger.info(f"Game events: {len(df_game)}")
    
    # Infer Team Name
    home_id = df_game['home_id'].iloc[0]
    away_id = df_game['away_id'].iloc[0]
    home_abb = df_game['home_abb'].iloc[0]
    away_abb = df_game['away_abb'].iloc[0]
    
    team_map = {home_id: home_abb, away_id: away_abb}
    df_game['team'] = df_game['team_id'].map(team_map)

    # 3. PREPARE DATA FOR MODELS
    # Nested needs imputation
    logger.info("Imputing for Nested Model...")
    df_game_imputed = impute.impute_blocked_shot_origins(df_game, method='point_pull')
    
    # 4. PREDICTIONS
    
    # Initialize columns
    df_game['xg_single'] = 0.0
    df_game['xg_nested'] = 0.0
    df_game['layer_prob_blocked'] = np.nan
    df_game['layer_prob_unblocked'] = np.nan
    df_game['layer_prob_on_net'] = np.nan
    df_game['layer_prob_finish'] = np.nan
    df_game['is_blocked'] = (df_game['event'] == 'blocked-shot').astype(int)
    
    # Valid shot events
    shot_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    mask_shots = df_game['event'].isin(shot_events)
    
    if mask_shots.any():
        logger.info(f"Predicting on {mask_shots.sum()} shot events...")
        df_shots = df_game.loc[mask_shots].copy()
        
        # A. Single Model xG
        try:
            # Prepare data using helper to match training encoding
            df_curr_single_ready, final_feats, _ = fit_xgs.clean_df_for_model(
                df_shots.copy(),                # IMPORTANT: clean_df_for_model returns feature columns + 'is_goal' target.
                feature_cols=raw_features_single,
                fixed_categorical_levels=cat_map_single
            )
            
            # Predict
            # We strictlty select features for prediction.
            if hasattr(clf_single, 'predict_proba'):
                probs_s_subset = clf_single.predict_proba(df_curr_single_ready[final_feats])[:, 1]
                
                # Map back to df_game
                # df_curr_single_ready.index should be subset of df_shots.index
                df_game.loc[df_curr_single_ready.index, 'xg_single'] = probs_s_subset
                
                # FORCE ZERO xG FOR BLOCKED SHOTS (Single Model Only)
                # The wrapper logic might be skipped if 'event' col is missing from input.
                # So we enforce it here on the results.
                df_game.loc[df_game['event'] == 'blocked-shot', 'xg_single'] = 0.0
            
        except Exception as e:
            logger.error(f"Single model prediction failed details: {e}")

        # B. Nested Model xG & Layers
        df_shots_imp = df_game_imputed.loc[mask_shots].copy()
        
        try:
            probs_nested = clf_nested.predict_proba(df_shots_imp)[:, 1]
            df_game.loc[mask_shots, 'xg_nested'] = probs_nested
            
            # Layers
            # Encode manually using stored encoders
            df_encoded = df_shots_imp.copy()
            if hasattr(clf_nested, 'le_shot'):
                 shot_map = dict(zip(clf_nested.le_shot.classes_, clf_nested.le_shot.transform(clf_nested.le_shot.classes_)))
                 fallback = shot_map.get('Unknown', 0)
                 df_encoded['shot_type_encoded'] = df_encoded['shot_type'].astype(str).map(shot_map).fillna(fallback).astype(int)
            
            if hasattr(clf_nested, 'le_state'):
                 state_map = dict(zip(clf_nested.le_state.classes_, clf_nested.le_state.transform(clf_nested.le_state.classes_)))
                 fallback_state = state_map.get('5v5', 0)
                 df_encoded['game_state_encoded'] = df_encoded['game_state'].astype(str).map(state_map).fillna(fallback_state).astype(int)
            
            # 1. Block
            p_blk = clf_nested.model_block.predict_proba(df_encoded[clf_nested.config_block.feature_cols])[:, 1]
            df_game.loc[mask_shots, 'layer_prob_blocked'] = p_blk
            df_game.loc[mask_shots, 'layer_prob_unblocked'] = 1.0 - p_blk
            
            # 2. Accuracy
            p_acc = clf_nested.model_accuracy.predict_proba(df_encoded[clf_nested.config_accuracy.feature_cols])[:, 1]
            df_game.loc[mask_shots, 'layer_prob_on_net'] = p_acc
            
            # 3. Finish
            p_fin = clf_nested.model_finish.predict_proba(df_encoded[clf_nested.config_finish.feature_cols])[:, 1]
            df_game.loc[mask_shots, 'layer_prob_finish'] = p_fin
            
        except Exception as e:
            logger.error(f"Nested model prediction failed: {e}")

    # 5. Save Output
    cols_to_save = [
        'game_id', 'period', 'period_time', 
        'event', 'team', 'player_name', 'shot_type', 
        'xg_single', 'xg_nested', 
        'layer_prob_blocked', 'layer_prob_unblocked', 'layer_prob_on_net', 'layer_prob_finish',
        'x', 'y', 'distance', 'angle_deg', 'is_blocked', 'game_state'
    ]
    
    final_cols = [c for c in cols_to_save if c in df_game.columns]
    remaining = [c for c in df_game.columns if c not in final_cols]
    output_columns = final_cols + remaining
    
    out_file = f"analysis/flyers_game_{last_game_id}_predictions.csv"
    df_game[output_columns].to_csv(out_file, index=False)
    
    logger.info(f"Predictions saved to {out_file}")
    print(f"File generated: {out_file}")

    # 6. Plot Comparison
    plot_file = f"analysis/flyers_game_{last_game_id}_xg_comparison.png"
    plt.figure(figsize=(10, 8))
    
    # Filter for valid shots that have predictions
    # We want to see how they differ.
    plot_df = df_game[mask_shots].copy()
    
    # Scatter plot
    # Color by event type
    events = plot_df['event'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(events)))
    
    for event, color in zip(events, colors):
        subset = plot_df[plot_df['event'] == event]
        plt.scatter(
            subset['xg_single'], 
            subset['xg_nested'], 
            alpha=0.7, 
            label=event,
            color=color,
            edgecolors='k',
            s=80
        )
        
    # Add identity line
    max_val = max(plot_df['xg_single'].max(), plot_df['xg_nested'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Identity (y=x)')
    
    plt.title(f"xG Model Comparison: Single vs Nested\nGame {last_game_id}")
    plt.xlabel("Single Model xG")
    plt.ylabel("Nested Model xG")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(plot_file, dpi=150)
    plt.close()
    
    logger.info(f"Comparison plot saved to {plot_file}")
    print(f"Plot generated: {plot_file}")

if __name__ == "__main__":
    main()
