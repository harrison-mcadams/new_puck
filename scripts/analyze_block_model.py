
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from puck import fit_xgboost_nested, fit_xgs, analyze, correction, impute, config

def analyze_block_model():
    print("--- Analyzing Block Model Performance & Logic ---")
    
    # 1. Load Data (Most recent season for "current behavior", + one old season for contrast?)
    # Let's use 2024-2025 (current) and 2018-2019 (historical baseline)
    seasons_to_check = ['2018-2019', '2024-2025']
    
    # Load Model (to get Block Model)
    model_path = 'analysis/xgs/xg_model_nested_all.joblib'
    print(f"Loading Nested Model from {model_path}...")
    model = joblib.load(model_path)
    
    # We need to access the block model directly. 
    # Wrapper doesn't expose it easily for predict_proba on specific DFs unless we access .model_block
    if not hasattr(model, 'model_block'):
        print("Error: Model does not have .model_block attribute. Is it the right class?")
        return

    # Helper to load season
    def load_season(s_name):
        base_dir = Path(config.DATA_DIR)
        # Try finding it
        for item in base_dir.iterdir():
            if item.name.replace('-','') == s_name.replace('-',''):
                csv_path = item / f"{item.name}_df.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    df = correction.fix_blocked_shot_attribution(df)
                    # We need shots + blocked shots
                    mask = df['event'].isin(['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot'])
                    return df[mask].copy()
        return None

    results = []
    
    for s_name in seasons_to_check:
        print(f"\nProcessing {s_name}...")
        df = load_season(s_name)
        if df is None:
            print(f"  Skipping {s_name} (Not found)")
            continue
            
        # Preprocess features using model's pipeline logic
        # Must replicate fit_xgboost_nested.preprocess_data roughly?
        # Actually better to use the model's own helper if possible, or replicate it.
        # The model needs 'distance', 'angle_deg', etc.
        
        # We need to impute blocked shot origins to know "where it came from" for the model to predict
        # This is CRITICAL: The model predicts P(Block) based on ORIGIN.
        # But actual blocked shots in raw data have coordinates at the BLOCK location (unless fixed).
        # We applied correction.fix_blocked_shot_attribution, but we ALSO need impute.
        try:
             df = impute.impute_blocked_shot_origins(df, method='point_pull')
        except: pass
        
        # Enrich Bios (shoots_catches)
        print("  Enriching Bios...")
        df = fit_xgs.enrich_data_with_bios(df)
        
        # Prepare for prediction
        # We'll use the model's features (excluding shot_type for block model)
        feat_block = [f for f in model.features if 'shot_type' not in f]
        
        # We need to ensure cols exist
        df_prep = fit_xgboost_nested.preprocess_data(df, features=model.features)
        
        # Predict
        # We probably need to handle categorical casting carefully like fit script
        # Quick hack: use model._prepare_df if available? No, it's internal.
        # We will trust our manual prep or try to use predict_proba partially?
        
        # Actually, let's just strip 'shot_type' and pass to model_block directly
        # Check dtypes
        for col in feat_block:
            if df_prep[col].dtype == 'object':
                df_prep[col] = df_prep[col].astype('category')
        
        try:
            probs = model.model_block.predict_proba(df_prep[feat_block])[:, 1]
        except Exception as e:
            print(f"Prediction failed: {e}")
            # Try fixing categoricals again
            # maybe vocab mismatch?
            continue
            
        df['prob_block'] = probs
        df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
        
        # Metrics
        auc = roc_auc_score(df['is_blocked'], probs)
        ll = log_loss(df['is_blocked'], probs)
        block_rate_actual = df['is_blocked'].mean()
        block_rate_pred = probs.mean()
        
        print(f"  AUC: {auc:.4f}")
        print(f"  LogLoss: {ll:.4f}")
        print(f"  Block Rate: {block_rate_actual:.2%} (Actual) vs {block_rate_pred:.2%} (Pred)")
        
        # Spatial Binning
        # Create x_fixed if missing (Simple absolute for one-zone view)
        if 'x_fixed' not in df.columns:
            df['x_fixed'] = df['x'].abs()
        if 'y_fixed' not in df.columns:
            # Flip Y if X was negative to maintain handedness relative to net? 
            # Or just raw Y. For blocking, maybe raw Y is fine?
            # Standard: if we flipped X, we flip Y usually.
            # But let's just use raw Y for now, or abs(Y) if we want quadrant?
            # Let's simple use Y.
            df['y_fixed'] = df['y']
            
        df['x_bin'] = (df['x_fixed'] // 10) * 10
        df['y_bin'] = (df['y_fixed'] // 10) * 10
        
        spatial = df.groupby(['x_bin', 'y_bin']).agg({
            'is_blocked': 'mean',
            'prob_block': 'mean',
            'event': 'count'
        }).reset_index()
        
        spatial = spatial[spatial['event'] > 50] # Min sample
        spatial['diff'] = spatial['prob_block'] - spatial['is_blocked']
        spatial['season'] = s_name
        results.append(spatial)

    # Plotting
    if not results: return
    
    all_spatial = pd.concat(results)
    
    # Plot Diff Maps
    g = sns.FacetGrid(all_spatial, col="season", height=5)
    def scatter_heatmap(data, color):
        # We want X, Y, Color=Diff
        # Use scatter
        plt.scatter(data['x_bin'], data['y_bin'], c=data['diff'], cmap='coolwarm', vmin=-0.2, vmax=0.2, s=200, marker='s')
        plt.colorbar(label='Pred - Actual (Red = Overpredict Block)')
        
    g.map_dataframe(scatter_heatmap)
    plt.savefig('analysis/block_model_spatial_diff.png')
    print("Saved analysis/block_model_spatial_diff.png")

if __name__ == "__main__":
    analyze_block_model()
