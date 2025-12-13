
import sys
import os
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import fit_xgs, fit_nested_xgs, impute

# Setup Plotting Style
plt.style.use('ggplot')
sns.set_palette("tab10")

OUT_DIR = Path('analysis/model_details')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_models():
    print("Loading models...")
    try:
        clf_single = joblib.load('analysis/xgs/xg_model_single.joblib')
        clf_nested = joblib.load('analysis/xgs/xg_model_nested.joblib')
        return clf_single, clf_nested
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

def load_data(season=20252026):
    path = Path(f'data/{season}/{season}_df.csv')
    if not path.exists():
        # Try fallback to 24-25 if 25-26 missing (though user context implies 25-26 exists)
        path = Path('data/20242025/20242025_df.csv')
    
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    return df

def plot_feature_importance(name, clf, feature_names, filename):
    """Plot feature importance for a generic Random Forest and save to filename."""
    if not hasattr(clf, 'feature_importances_'):
        print(f"Model {name} does not have feature_importances_")
        return

    importances = clf.feature_importances_
    indices = np.argsort(importances) # Sort ascending

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances: {name}")
    plt.barh(range(len(indices)), importances[indices], align='center')
    
    # Map indices to names
    sorted_names = [feature_names[i] for i in indices]
    plt.yticks(range(len(indices)), sorted_names)
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    
    out_path = OUT_DIR / filename
    plt.savefig(out_path)
    plt.close()
    print(f"Saved importance plot to {out_path}")

def analyze_feature_importance(clf_single, clf_nested):
    print("\n--- Feature Importance Analysis ---")
    
    # 1. Single Model
    # Need feature names. Usually stored in metadata, or we can infer.
    # We'll try to load metadata file.
    try:
        with open('analysis/xgs/xg_model_single.joblib.meta.json', 'r') as f:
            meta = json.load(f)
            # 'final_features' in metadata usually contains the expanded one-hot names
            feats_single = meta.get('final_features')
    except:
        feats_single = None

    # Unwrap if implicit wrapper
    model_s = clf_single.clf if isinstance(clf_single, fit_xgs.SingleXGClassifier) else clf_single
    
    if feats_single and hasattr(model_s, 'feature_importances_'):
        if len(feats_single) == len(model_s.feature_importances_):
            plot_feature_importance("Single Model", model_s, feats_single, "importance_single.png")
        else:
             print(f"Mismatch in single model features: Meta {len(feats_single)} vs Model {len(model_s.feature_importances_)}")

    # 2. Nested Model (3 Layers)
    # NestedXGClassifier has: .model_block, .model_accuracy, .model_finish
    # And configs: .config_block.feature_cols, etc.
    # Note: Internal models use ENCODED features (e.g., 'shot_type_encoded'). 
    # The config feature_cols listed in __init__ might be the raw ones or the ones passed to predict.
    # Let's check the encoded logic in NestedXGClassifier.
    # Actually, in my refactor, I pass specific cols to the internal models.
    # Let's trust the feature_importances_ length matches the training cols.
    
    # We need the conceptual names.
    # Block Model: ['distance', 'angle_deg', 'is_net_empty', 'game_state_encoded'] (from fit_nested_xgs.py __init__)
    # Let's define them manually based on known architecture if dynamic lookup fails.
    
    def analyze_layer(layer_name, model, expected_feats):
         if hasattr(model, 'feature_importances_'):
             # If encoding used (e.g. shot_type -> shot_type_encoded), the model sees 1 col per feature
             # largely because we used LabelEncoding/integer coding, NOT one-hot.
             # So feature count should match logical count.
             if len(expected_feats) == len(model.feature_importances_):
                 plot_feature_importance(f"Nested: {layer_name}", model, expected_feats, f"importance_nested_{layer_name.lower().replace(' ','_')}.png")
             else:
                 print(f"Mismatch in {layer_name}: Expected {expected_feats}, got {len(model.feature_importances_)}")

    analyze_layer("Block Model", clf_nested.model_block, clf_nested.config_block.feature_cols)
    analyze_layer("Accuracy Model", clf_nested.model_accuracy, clf_nested.config_accuracy.feature_cols)
    analyze_layer("Finish Model", clf_nested.model_finish, clf_nested.config_finish.feature_cols)


def plot_continuous_effect(model, df, feature, name, ax):
    """Plot partial dependence-like effect for continuous feature."""
    # Bin feature
    df['bin'] = pd.cut(df[feature], bins=20)
    grouped = df.groupby('bin')[feature].mean()
    
    # Predict for each bin (marginal effect approach approx)
    # Actually, simpler: just plot mean prediction vs bin center
    # This shows correlation, not causal PDP, but usually enough for interpretation
    
    # Check if model has simple predict_proba
    # We need to construct a "test" set where we vary 'feature' and hold others constant?
    # Or just plotting the observed relationship (Model xG vs Feature) is what user wants "how do different options compare"
    
    # Let's do Observed Average xG vs Feature Value
    y_col = 'xg_single' if 'single' in name.lower() else 'xg_nested'
    
    # We need predictions in df
    # Assuming df has predictions already? No, we need to generate them or pass them.
    pass

def analyze_feature_effects(clf_single, clf_nested, df):
    print("\n--- Feature Effects Analysis ---")
    
    # 1. Generate Predictions for specific features to analyze
    # We want to see how xG varies with Feature X, all else equal (PDP) OR just average (Marginal)
    # Let's do Marginal (Observed) first as it's faster and shows "what usually happens".
    
    # We need to add predictions to df if not present
    # But strictly for the "Unblocked" subset for Single Model comparison?
    # User wants model comparison.
    
    # Let's focus on Unblocked shots for fair comparison
    df_unblocked = df[df['event'] != 'blocked-shot'].copy()
    
    # Impute for nested
    df_imp = impute.impute_blocked_shot_origins(df_unblocked, method='mean_6')
    
    # Predict Single
    # We must clean/encode first using metadata logic
    # Try to load metadata
    raw_features_single = ['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type']
    cat_map_single = {}
    try:
        with open('analysis/xgs/xg_model_single.joblib.meta.json', 'r') as f:
            meta_single = json.load(f)
            cat_map_single = meta_single.get('categorical_levels_map', {})
    except:
        pass
        
    df_s_ready, feats_s, _ = fit_xgs.clean_df_for_model(
        df_unblocked.copy(), 
        feature_cols=raw_features_single,
        fixed_categorical_levels=cat_map_single
    )
    
    # Predict (Selecting only features)
    probs_s = clf_single.predict_proba(df_s_ready[feats_s])[:, 1]
    
    # Map back (df_s_ready indices match df_unblocked if no drops, but fit_xgs cleans NaNs)
    # We need to map by index
    df_unblocked.loc[df_s_ready.index, 'xg_single'] = probs_s
    
    # Predict Nested
    probs_n = clf_nested.predict_proba(df_imp)[:, 1]
    df_unblocked['xg_nested'] = probs_n

    # PLOT 1: Categorical Features (Bar Chart)
    # Shot Type
    plt.figure(figsize=(12, 5))
    
    # Group by shot_type
    # Melt for side-by-side
    df_melt = df_unblocked.melt(id_vars=['shot_type'], value_vars=['xg_single', 'xg_nested'], 
                                var_name='Model', value_name='xG')
    
    sns.barplot(data=df_melt, x='shot_type', y='xG', hue='Model')
    plt.title("Average xG by Shot Type (Unblocked Shots)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "effect_shot_type.png")
    plt.close()
    
    # Game State
    plt.figure(figsize=(12, 5))
    df_melt_gs = df_unblocked.melt(id_vars=['game_state'], value_vars=['xg_single', 'xg_nested'], 
                                var_name='Model', value_name='xG')
    
    # Order by frequency/relevance? Or just mean xG?
    order = df_unblocked.groupby('game_state')['xg_nested'].mean().sort_values(ascending=False).index[:10]
    sns.barplot(data=df_melt_gs, x='game_state', y='xG', hue='Model', order=order)
    plt.title("Average xG by Game State (Top 10)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "effect_game_state.png")
    plt.close()
    
    # PLOT 2: Continuous Features (Line Plot of Binned Means)
    # Distance
    df_unblocked['dist_bin'] = pd.cut(df_unblocked['distance'], bins=np.linspace(0, 100, 21))
    
    # Agg
    agg_dist = df_unblocked.groupby('dist_bin')[['xg_single', 'xg_nested']].mean().reset_index()
    agg_dist['dist_mid'] = agg_dist['dist_bin'].apply(lambda x: x.mid).astype(float)
    
    plt.figure(figsize=(10, 6))
    plt.plot(agg_dist['dist_mid'], agg_dist['xg_single'], marker='o', label='Single Model')
    plt.plot(agg_dist['dist_mid'], agg_dist['xg_nested'], marker='x', label='Nested Model')
    plt.xlabel("Distance (ft)")
    plt.ylabel("Average xG")
    plt.title("Average xG vs Distance")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUT_DIR / "effect_distance.png")
    plt.close()

    # Angle
    df_unblocked['angle_bin'] = pd.cut(df_unblocked['angle_deg'].abs(), bins=np.linspace(0, 100, 21)) # 0 to 100 degrees
    agg_angle = df_unblocked.groupby('angle_bin')[['xg_single', 'xg_nested']].mean().reset_index()
    agg_angle['angle_mid'] = agg_angle['angle_bin'].apply(lambda x: x.mid).astype(float)
    
    plt.figure(figsize=(10, 6))
    plt.plot(agg_angle['angle_mid'], agg_angle['xg_single'], marker='o', label='Single Model')
    plt.plot(agg_angle['angle_mid'], agg_angle['xg_nested'], marker='x', label='Nested Model')
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Average xG")
    plt.title("Average xG vs Angle")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUT_DIR / "effect_angle.png")
    plt.close()
    
    print("Saved feature effect plots.")


def main():
    clf_single, clf_nested = load_models()
    df = load_data()
    
    # Filter only shots for analysis
    mask_shots = df['event'].isin(['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal'])
    df_shots = df[mask_shots].copy()
    
    # 1. Feature Importance
    analyze_feature_importance(clf_single, clf_nested)
    
def analyze_game_calibration(clf_single, clf_nested, df):
    print("\n--- Game Calibration Analysis ---")
    
    # We need predictions on the FULL dataset (including blocks for Nested, excluding for Single)
    # Single Model Logic: xG = 0 for blocks.
    # Nested Model Logic: xG > 0 for blocks.
    
    # Prepare
    mask_events = df['event'].isin(['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal'])
    df_play = df[mask_events].copy()
    
    # Impute
    df_imp = impute.impute_blocked_shot_origins(df_play, method='mean_6')
    
    # Predict Single
    # We must clean/encode first using metadata logic
    # Try to load metadata
    raw_features_single = ['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type']
    cat_map_single = {}
    try:
        with open('analysis/xgs/xg_model_single.joblib.meta.json', 'r') as f:
            meta_single = json.load(f)
            cat_map_single = meta_single.get('categorical_levels_map', {})
    except:
        pass
        
    df_s_ready, feats_s, _ = fit_xgs.clean_df_for_model(
        df_play.copy(), 
        feature_cols=raw_features_single,
        fixed_categorical_levels=cat_map_single
    )
    
    # Predict (Selecting only features)
    probs_s = clf_single.predict_proba(df_s_ready[feats_s])[:, 1]
    
    # Map back (df_s_ready indices match df_play if no drops, but fit_xgs cleans NaNs)
    df_play.loc[df_s_ready.index, 'xg_single'] = probs_s
    
    # Enforce 0 for blocked explicitly
    df_play.loc[df_play['event'] == 'blocked-shot', 'xg_single'] = 0.0
    
    print(f"Single Model xG Stats: Mean={df_play['xg_single'].mean():.4f}, Max={df_play['xg_single'].max():.4f}, Sum={df_play['xg_single'].sum():.4f}")
        
    # Predict Nested
    df_play['xg_nested'] = clf_nested.predict_proba(df_imp)[:, 1]
    
    # Aggregate per Game
    # Assuming 'game_id' exists
    # Goals: event == 'goal'
    df_play['is_goal'] = (df_play['event'] == 'goal').astype(int)
    
    game_agg = df_play.groupby('game_id').agg({
        'is_goal': 'sum',
        'xg_single': 'sum',
        'xg_nested': 'sum'
    }).reset_index()
    
    game_agg.rename(columns={'is_goal': 'Actual Goals'}, inplace=True)
    
    # Plot Scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(game_agg['xg_single'], game_agg['Actual Goals'], alpha=0.5, label='Single Model')
    plt.scatter(game_agg['xg_nested'], game_agg['Actual Goals'], alpha=0.5, label='Nested Model', marker='x')
    
    # Identity line
    max_val = max(game_agg['Actual Goals'].max(), game_agg['xg_single'].max(), game_agg['xg_nested'].max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    
    plt.xlabel("Predicted Goals (xG)")
    plt.ylabel("Actual Goals")
    plt.title("Game Calibration: Predicted vs Actual Goals (2025-2026)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "calibration_game_scatter.png")
    plt.close()
    
    # Calculate Correlations
    corr_s = game_agg['xg_single'].corr(game_agg['Actual Goals'])
    corr_n = game_agg['xg_nested'].corr(game_agg['Actual Goals'])
    print(f"Goal Correlation per Game:")
    print(f"  Single: {corr_s:.3f}")
    print(f"  Nested: {corr_n:.3f}")


def main():
    clf_single, clf_nested = load_models()
    df = load_data()
    
    # Filter only shots for analysis
    mask_shots = df['event'].isin(['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal'])
    df_shots = df[mask_shots].copy()
    
    # 1. Feature Importance
    analyze_feature_importance(clf_single, clf_nested)
    
    # 2. Feature Effects
    analyze_feature_effects(clf_single, clf_nested, df_shots)
    
    # 3. Game Calibration (Using full df to catch all events/games)
    analyze_game_calibration(clf_single, clf_nested, df)
    
    print("\nDone. Results saved to analysis/model_details/")

if __name__ == "__main__":
    main()
