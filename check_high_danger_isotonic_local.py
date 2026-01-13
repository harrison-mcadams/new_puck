
import pandas as pd
import numpy as np

def main():
    print("Loading model...")
    try:
        import joblib
        from puck.fit_xgboost_nested import XGBNestedXGClassifier
        model = joblib.load('analysis/xgs/xg_model_nested.joblib')
        
        # Backward compatibility patch if needed
        if not hasattr(model, 'categorical_priors_'):
            model.categorical_priors_ = {}
            if hasattr(model, 'shot_type_priors_') and model.shot_type_priors_:
                model.categorical_priors_['shot_type'] = model.shot_type_priors_
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Loading comparison data...")
    try:
        # Use deep dive which has features
        df = pd.read_csv('analysis/shot_comparison_deep_dive_2025.csv')
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
        
    # Recalculate xgs with new model
    print("Recalculating predictions with Isotonic Model...")
    
    # Handle missing game_state if necessary
    if 'game_state' not in df.columns:
        df['game_state'] = '5v5'
    else:
        df['game_state'] = df['game_state'].fillna('5v5')
        
    df['xgs'] = model.predict_proba(df)[:, 1]


    print("\n--- Max xG Values ---")
    print(f"My Model (Isotonic) Max: {df['xgs'].max():.4f}")
    print(f"MoneyPuck Max:           {df['xGoal'].max():.4f}")

    print("\n--- High Danger Distribution (Count & %) ---")
    thresholds = [0.3, 0.5, 0.7, 0.9]
    for t in thresholds:
        my_count = len(df[df['xgs'] > t])
        mp_count = len(df[df['xGoal'] > t])
        print(f"xG > {t}: My={my_count:<5} ({my_count/len(df):.1%}) | MP={mp_count:<5} ({mp_count/len(df):.1%})")

    print("\n--- Analysis of 'High Danger' Shots (MoneyPuck > 0.5) ---")
    # Filter for shots that MoneyPuck considers high danger
    high_danger_mp = df[df['xGoal'] > 0.5]
    print(f"Count of MP > 0.5: {len(high_danger_mp)}")
    
    if len(high_danger_mp) > 0:
        print(f"My Model Mean on these: {high_danger_mp['xgs'].mean():.4f}")
        print(f"My Model Max on these:  {high_danger_mp['xgs'].max():.4f}")
        print(f"My Model > 0.5 count:   {len(high_danger_mp[high_danger_mp['xgs'] > 0.5])}")
        
        # Check for extreme disconnects
        disconnects = high_danger_mp[high_danger_mp['xgs'] < 0.2]
        print(f"\nSevere Disconnects (MP > 0.5, My < 0.2): {len(disconnects)}")
        if len(disconnects) > 0:
            print("Examples:")
            print(disconnects[['game_id', 'event', 'xgs', 'xGoal', 'distance', 'shotType']].head(5))

if __name__ == "__main__":
    main()
