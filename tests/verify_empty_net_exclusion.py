import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck.fit_xgs import clean_df_for_model
from puck.fit_nested_xgs import NestedXGClassifier

def test_exclusion():
    print("Testing Empty Net Exclusion...")

    # Create dummy data
    data = {
        'event': ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot', 'shot-on-goal'],
        'distance': [10, 20, 30, 40, 150],
        'angle_deg': [0, 10, 5, 2, 0],
        'is_net_empty': [0, 0, 0, 0, 1], # Last one is empty net
        'game_state': ['5v5'] * 5,
        'shot_type': ['Wrist'] * 5
    }
    df = pd.DataFrame(data)
    
    print(f"Original Row Count: {len(df)}")
    print(f"Empty Net Count: {df['is_net_empty'].sum()}")

    # Test fit_xgs.clean_df_for_model
    print("\n--- Testing fit_xgs.clean_df_for_model ---")
    df_clean, _, _ = clean_df_for_model(df, feature_cols=['distance', 'angle_deg', 'is_net_empty'])
    print(f"Cleaned Row Count: {len(df_clean)}")
    
    # Assert
    assert len(df_clean) == 4, f"Expected 4 rows, got {len(df_clean)}"
    if 'is_net_empty' in df_clean.columns:
         assert df_clean['is_net_empty'].sum() == 0, "Found empty net shots in cleaned data!"
    print("PASS: fit_xgs.clean_df_for_model filtered correctly.")

    # Test fit_nested_xgs logic (manually invoking preprocess_features logic logic since it's internal to fit but we can check the class)
    # Actually, we can just instantiate the class and call fit?
    # Or import the preprocess logic if we exposed it? We didn't expose it, it's inside main() or fit().
    # Wait, in fit_nested_xgs.py, `preprocess_features` is a standalone function at module level!
    
    from puck.fit_nested_xgs import preprocess_features
    print("\n--- Testing fit_nested_xgs.preprocess_features ---")
    
    # Reset df
    df = pd.DataFrame(data)
    df_pre = preprocess_features(df)
    
    print(f"Preprocessed Row Count: {len(df_pre)}")
    
    # Assert
    assert len(df_pre) == 4, f"Expected 4 rows, got {len(df_pre)}"
    assert df_pre['is_net_empty'].sum() == 0, "Found empty net shots in preprocessed data!"
    print("PASS: fit_nested_xgs.preprocess_features filtered correctly.")

if __name__ == "__main__":
    try:
        test_exclusion()
        print("\nALL TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
