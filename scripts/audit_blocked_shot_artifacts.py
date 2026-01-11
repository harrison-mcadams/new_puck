
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import data_pipeline, features, enrich

def main():
    print("Loading data...")
    # Load 2024-2025 samples to be representative
    # Ideally use a large file. 
    # If standard file is available use it.
    fpath = "data/20242025/20242025_df.csv"
    if not os.path.exists(fpath):
        # Fallback to creating it if possible or using what we have.
        # Try to find a recent large csv
        possible = [f for f in os.listdir("data/raw") if f.endswith(".csv") and "2024" in f]
        if possible:
            fpath = os.path.join("data/raw", possible[0])
        else:
            print("No data found.")
            return

    print(f"Reading {fpath}...")
    df = pd.read_csv(fpath)
    
    # Enrich
    print("Enriching data...")
    enricher = enrich.PlayerEnricher()
    df = enricher.enrich_dataframe(df)
    
    # Sample if too huge (keep it fast for audit)
    if len(df) > 200000:
        df = df.sample(200000, random_state=42)
        
    # Pipeline
    print("Running Pipeline (Processing)...")
    # We want to audit the output that goes into the XGBoost model.
    # So apply adjustments, imputation, etc.
    # We DO NOT want filtering yet, because we need to define our own target.
    # But wait, pipeline filters out blocks? No, standard pipeline keeps them.
    df = data_pipeline.preprocess_features(df, apply_imputation=True, apply_arena_adjustments=True, apply_dithering=True)
    
    # Filter to Shots vs Blocks
    valid_events = ['shot-on-goal', 'missed-shot', 'goal', 'blocked-shot']
    df = df[df['event'].isin(valid_events)].copy()
    
    # Target: 1 if Blocked, 0 otherwise
    df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
    
    print(f"Dataset: {len(df)} rows. Blocked: {df['is_blocked'].sum()} ({df['is_blocked'].mean():.1%})")
    
    # Feature Engineering (Adversarial)
    # We create features to test specific hypotheses about artifacts.
    
    # 1. Coordinate Precision Artifacts on X
    # (Do coords end in .0 or .5 or .1234?)
    # Calculate fractional part
    df['x_frac'] = df['x'] % 1.0
    df['y_frac'] = df['y'] % 1.0
    
    # 2. Standard Features
    # Format them exactly as model would see them
    df = data_pipeline._format_features(df)
    
    # Features to Audit
    # We include 'x_frac', 'y_frac' to detect float vs int artifacts
    feature_cols = [
        'x', 'y', 'distance', 'angle_deg', 
        'is_rebound', 'is_rush', 
        'time_elapsed_in_period_s', 'score_diff',
        'x_frac', 'y_frac'
    ]
    
    X = df[feature_cols].copy()
    y = df['is_blocked']
    
    # Handle Categoricals
    for c in X.select_dtypes(include=['object', 'category']).columns:
        X[c] = X[c].astype('category')
        
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train Detective Model
    print("Training Adversarial Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        enable_categorical=True,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"\nAdversarial AUC: {auc:.4f}")
    
    if auc > 0.7:
        print("!! WARNING: High discriminability detected. Model can easily distinguish Blocks !!")
    else:
        print("Discriminability is moderate. Natural differences likely dominate.")

    # Feature Importance
    imps = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop Discriminating Features:")
    print(imps.head(10))
    
    # Detailed check on Top 1
    top_feat = imps.iloc[0]['feature']
    print(f"\nAnalyzing Top Feature: {top_feat}")
    
    if top_feat in ['x_frac', 'y_frac']:
        print("  -> Precision Artifact Detected! Imputed coords likely have different precision than raw.")
        print("  Stats (Mean Frac):")
        print(df.groupby('is_blocked')[top_feat].mean())
    elif top_feat == 'shot_type':
        print("  -> Shot Type Leakage! Distribution differs significantly.")
        print(pd.crosstab(df['shot_type'], df['is_blocked'], normalize='columns'))
        
    # Save Feature Importance Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imps.head(10), x='importance', y='feature')
    plt.title(f'Adversarial Validation: Top Artifacts (AUC={auc:.2f})')
    plt.tight_layout()
    plt.savefig('analysis/audit_blocked_artifacts.png')
    print("\nSaved plot to analysis/audit_blocked_artifacts.png")

if __name__ == "__main__":
    main()
