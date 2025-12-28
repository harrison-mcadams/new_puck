import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sys
import os

# --- 1. Imputation Strategies ---

def apply_projection(df_in, dist_func):
    """Helper to apply back-projection logic for BLOCKED shots only."""
    df = df_in.copy()
    mask = df['event'] == 'blocked-shot'
    
    # Get coordinates
    bx = df.loc[mask, 'x']
    by = df.loc[mask, 'y']
    
    # Net Location
    net_x = np.where(bx > 0, 89, -89)
    net_y = 0
    
    vx = bx - net_x
    vy = by - net_y
    mag = np.sqrt(vx**2 + vy**2) # Distance from Net
    
    ux = vx / mag
    uy = vy / mag
    ux = ux.fillna(0)
    uy = uy.fillna(0)
    
    # Calculate Projection
    d_proj = dist_func(mag)
    
    # Project
    ox = bx + (ux * d_proj)
    oy = by + (uy * d_proj)
    
    # Clamp to Rink
    ox = np.clip(ox, -99.0, 99.0)
    oy = np.clip(oy, -42.0, 42.0)
    
    # Update
    df.loc[mask, 'imputed_x'] = ox
    df.loc[mask, 'imputed_y'] = oy
    
    # Recalculate Distance (Simple Euclidean)
    # Note: We aren't doing the full angle update here for speed, just distance
    new_dist = np.sqrt((ox - net_x)**2 + (oy - net_y)**2)
    df.loc[mask, 'distance'] = new_dist # Update the column used for modeling
    
    return df

def strat_baseline(df):
    return apply_projection(df, lambda mag: 5.64)

def strat_crease_fix(df):
    """Smart: If < 15ft, +25ft"""
    return apply_projection(df, lambda mag: np.where(mag < 15.0, 25.0, 5.64))

def strat_smooth_point(df):
    """Smooth Point: If < 30ft, target ~ Normal(55, 8)"""
    def dist_logic(mag):
        targets = np.random.normal(loc=55.0, scale=8.0, size=len(mag))
        d_raw = targets - mag
        d_proj = np.maximum(5.64, d_raw)
        return np.where(mag < 30.0, d_proj, 5.64)
    return apply_projection(df, dist_logic)

# --- 2. Evaluation Logic ---

def train_block_model(df_imputed, label):
    """
    Train a simple Block Model (Is Blocked?) on the imputed data.
    We want to see if the model learns 'Close = Safe' or gets confused.
    """
    # Features: Just distance for this pure test? Or Dist + Angle?
    # Let's use Dist + Angle to be realistic
    X = df_imputed[['distance', 'angle_deg']]
    y = df_imputed['event'] == 'blocked-shot'
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    clf.fit(X_train, y_train)
    
    probs = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    
    return clf, auc

# --- MAIN ---

if __name__ == "__main__":
    # Add current dir to path
    sys.path.append(os.getcwd())
    
    # 1. Load Data
    df = None
    try:
        from puck import fit_nested_xgs, fit_xgboost_nested
        print("Loading raw data...")
        df = fit_nested_xgs.load_data()
        df = fit_xgboost_nested.preprocess_data(df)
    except Exception as e:
        print(f"Module load failed: {e}")
        # Fallback to CSV
        paths = ['analysis/nested_xgs/test_predictions.csv', 'test_predictions.csv'] 
        for p in paths:
            if os.path.exists(p):
                df = pd.read_csv(p)
                break
    
    if df is None:
        print("No data found.")
        exit(1)
        
    print(f"Data Loaded: {len(df)} rows")
    
    # 2. Prepare Reference Distributions
    mask_on_net = df['event'].isin(['shot-on-goal', 'goal'])
    dist_on_net = df.loc[mask_on_net, 'distance']
    
    mask_blocked = df['event'] == 'blocked-shot'
    dist_raw_blocked = df.loc[mask_blocked, 'distance'] # Pre-imputation
    
    print(f"Shots on Net: {len(dist_on_net)}")
    print(f"Blocked Shots: {len(dist_raw_blocked)}")
    
    # 3. Apply Strategies
    strategies = {
        'Baseline': strat_baseline,
        'Smart (+25)': strat_crease_fix,
        'Smooth Point': strat_smooth_point
    }
    
    results = {}
    models = {}
    
    plt.figure(figsize=(12, 8))
    
    # Plot References
    sns.kdeplot(dist_on_net, label='All Shots On Net (Ref)', color='black', linestyle='--', linewidth=2)
    sns.kdeplot(dist_raw_blocked, label='Raw Block Locs (Pre-Imp)', color='grey', shade=True, alpha=0.3)
    
    colors = ['red', 'orange', 'green']
    
    for (name, func), color in zip(strategies.items(), colors):
        print(f"\n--- Processing {name} ---")
        
        # A. Impute
        # We need to impute ONLY blocked shots, but keep non-blocked for the model training!
        # The strategy functions typically take the whole DF and update blocked.
        # Ensure we pass the WHOLE df.
        df_imp = func(df)
        
        # Stats
        dists = df_imp.loc[mask_blocked, 'distance']
        print(dists.describe().to_string())
        
        # B. Plot Distribution
        sns.kdeplot(dists, label=f'Imputed: {name}', color=color)
        
        # C. Train Model
        print(f"Training Block Model for {name}...")
        clf, auc = train_block_model(df_imp, name)
        print(f"AUC: {auc:.4f}")
        models[name] = clf
        
    plt.title("Imputed Blocked Shot Origins vs Reference Distributions")
    plt.xlabel("Distance (ft)")
    plt.xlim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('analysis/imputation_dist_comparison.png')
    print("\nSaved Dist Plot to analysis/imputation_dist_comparison.png")
    
    # 4. Plot Model Behavior (Prob Block vs Distance)
    plt.figure(figsize=(10, 6))
    
    # Generate a dummy range of shots
    # Fixed Angle (0 = Straight on), Varying Distance 0-100
    dummy_dist = np.linspace(0, 100, 100)
    X_dummy = pd.DataFrame({
        'distance': dummy_dist,
        'angle_deg': 0 
    })
    
    for name, clf in models.items():
        probs = clf.predict_proba(X_dummy)[:, 1]
        plt.plot(dummy_dist, probs, label=f'Model: {name}')
        
    plt.title("Block Model: Probability of Block vs Distance (Angle=0)")
    plt.xlabel("Distance (ft)")
    plt.ylabel("P(Blocked)")
    plt.legend()
    plt.grid(True)
    plt.savefig('analysis/imputation_model_behavior.png')
    print("Saved Model Plot to analysis/imputation_model_behavior.png")
