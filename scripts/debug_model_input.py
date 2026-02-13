
import os
import sys
import pandas as pd
import numpy as np
import joblib

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import mixed_effects
from puck import fit_nested_xgs
from puck import impute

def main():
    print("Loading model...")
    model_path = "analysis/xgs/xg_model_nested_tensor.joblib"
    try:
        base_model = joblib.load(model_path)
        print(f"Model type: {type(base_model)}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    expected_features = getattr(base_model, 'features', [])
    print(f"Expected features ({len(expected_features)}): {expected_features}")

    # Load small sample of data
    season = "20252026"
    paths = [
        os.path.join("data", season, f"{season}.csv"),
        os.path.join("data", f"{season}.csv"),
        os.path.join("data", "processed", season, f"{season}.csv")
    ]
    df = None
    for p in paths:
        if os.path.exists(p):
            print(f"Loading data from {p}...")
            df = pd.read_csv(p)
            break
            
    if df is None:
        print("No data found.")
        return


    # Preprocess using data_pipeline
    print("Preprocessing with coverage check...")
    from puck import data_pipeline
    df = data_pipeline.preprocess_features(
        df, 
        is_training=False,
        apply_imputation=True,
        apply_arena_adjustments=True,
        apply_bio_enrichment=True,
        apply_filtering=True
    )
    
    # Force clean Infs just in case pipeline doesn't
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], 0)

    print(f"Data shape after pipeline: {df.shape}")
    
    # Check missing columns
    expected_features = getattr(base_model, 'features', [])
    missing = [f for f in expected_features if f not in df.columns]
    print(f"Missing columns: {missing}")
    
    # Check NaNs/Infs in expected features
    print("\nCheck for bad values in features:")
    for f in expected_features:
        if f in df.columns:
            n_nan = df[f].isna().sum()
            n_inf = np.isinf(pd.to_numeric(df[f], errors='coerce')).sum()
            if n_nan > 0 or n_inf > 0:
                print(f"  {f}: NaNs={n_nan}, Infs={n_inf}")
    
    # Try prediction
    print("\nAttempting prediction on 10 rows...")
    try:
        sample = df.head(10).copy()
        preds = base_model.predict_proba(sample)
        print(f"Prediction success! Shape: {preds.shape}")
        print(preds)
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
