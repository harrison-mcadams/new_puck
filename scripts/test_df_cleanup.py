
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgs
from puck import features

def test_cleanup():
    print("Testing clean_df_for_model with new features...")
    
    # Create a dummy dataframe with the new columns
    data = {
        'event': ['shot-on-goal', 'missed-shot', 'goal'],
        'is_net_empty': [0, 0, 0],
        'game_state': ['5v5', '5v4', '5v5'],
        'score_diff': [0, 1, -1],
        'period_number': [1, 2, 3],
        'time_elapsed_in_period_s': [60, 120, 180],
        'total_time_elapsed_s': [60, 1320, 3780],
        'shot_type': ['Wrist Shot', 'Slap Shot', 'Snap Shot'],
        'distance': [10, 20, 5],
        'angle_deg': [0, 15, -5],
        'is_goal': [0, 0, 1]
    }
    df = pd.DataFrame(data)
    
    # Get the 'all_inclusive' feature set which should now include the new features
    feats = features.get_features('all_inclusive')
    print(f"Features: {feats}")
    
    # Run cleanup
    try:
        cleaned_df, final_feats, _ = fit_xgs.clean_df_for_model(df, feats)
        
        print("\nCleaned DataFrame Columns:")
        print(cleaned_df.columns.tolist())
        
        # Verify columns exist
        missing = [f for f in feats if f not in cleaned_df.columns]
        if missing:
            print(f"\nFAILED: Missing features in cleaned DF: {missing}")
            sys.exit(1)
            
        print("\nSUCCESS: All features preserved.")
        
    except Exception as e:
        print(f"\nFAILED: Exception during cleanup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_cleanup()
