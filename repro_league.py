
import sys
import pandas as pd
import parse

# Mock timing import as analyze.py does
try:
    import timing_new as timing
    sys.modules['timing'] = timing
    print(f"Imported timing_new as timing: {timing}")
except ImportError:
    import timing
    print(f"Imported timing (old): {timing}")

def run_repro():
    print("\n--- Checking load_season_df ---")
    if hasattr(timing, 'load_season_df'):
        print("timing.load_season_df exists")
    else:
        print("timing.load_season_df MISSING")
        # Try to find where it might be
        if hasattr(parse, '_season'):
            print("parse._season exists (likely candidate)")

    # Create a dummy dataframe to test build_mask
    print("\n--- Testing build_mask ---")
    df = pd.DataFrame({
        'game_state': ['5v5', '5v4', '4v5', '5v5', '3v3'],
        'x': [1, 2, 3, 4, 5],
        'y': [1, 2, 3, 4, 5]
    })
    print("Original DF:")
    print(df)

    condition = {'game_state': ['5v5']}
    print(f"Condition: {condition}")

    try:
        mask = parse.build_mask(df, condition)
        print(f"Mask: {mask.tolist()}")
        df_filtered = df.loc[mask]
        print("Filtered DF:")
        print(df_filtered)
        
        if len(df_filtered) == 2:
            print("SUCCESS: build_mask filtered correctly")
        else:
            print(f"FAILURE: Expected 2 rows, got {len(df_filtered)}")
    except Exception as e:
        print(f"FAILURE: build_mask raised exception: {e}")

    # Test with missing column
    print("\n--- Testing build_mask with missing column ---")
    df_missing = df.drop(columns=['game_state'])
    try:
        mask = parse.build_mask(df_missing, condition)
        print(f"Mask (missing col): {mask.tolist()}")
        if not mask.any():
            print("SUCCESS: build_mask returned all False for missing column")
        else:
            print("FAILURE: build_mask did not return all False")
    except Exception as e:
        print(f"FAILURE: build_mask raised exception: {e}")

if __name__ == '__main__':
    run_repro()
