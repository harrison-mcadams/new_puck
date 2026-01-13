
import pandas as pd
import numpy as np
import joblib

# Constants from your project structure - adapt as needed if imports fail
try:
    from puck.fit_xgboost_nested import XGBNestedXGClassifier
except ImportError:
    import sys
    sys.path.append('.')
    from puck.fit_xgboost_nested import XGBNestedXGClassifier

def main():
    print("Loading model...")
    try:
        model = joblib.load('analysis/xgs/xg_model_nested_all.joblib')
        # Backward compatibility patch
        if not hasattr(model, 'categorical_priors_'):
            print("Patching model: Adding missing categorical_priors_ attribute.")
            model.categorical_priors_ = {}
            if hasattr(model, 'shot_type_priors_') and model.shot_type_priors_:
                model.categorical_priors_['shot_type'] = model.shot_type_priors_
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Loading comparison data...")
    try:
        df = pd.read_csv('analysis/shot_comparison_deep_dive_2025.csv')
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
        
    # Handle missing game_state if necessary
    if 'game_state' not in df.columns:
        print("Warning: 'game_state' missing. Defaulting to '5v5'.")
        df['game_state'] = '5v5'
    else:
        # Fill NaNs in game_state just in case
        df['game_state'] = df['game_state'].fillna('5v5')
        
    # Filter for "High Danger" according to MoneyPuck (xG > 0.5)
    # This gives us a set of shots we "know" should be high value
    hd = df[df['xGoal'] > 0.5].copy()
    print(f"\nAnalyzing {len(hd)} MoneyPuck High Danger shots (MP > 0.5)...")
    
    # We need to re-run prediction to get layer outputs
    # Note: This assumes the dataframe 'df' has all necessary features for the model.
    # If using 'shot_comparison_2025.csv', it might have features, or we might need to load the full dataset.
    # Let's check columns first.
    required_cols = ['distance', 'angle_deg', 'shot_type', 'is_rebound', 'game_state']
    missing_cols = [c for c in required_cols if c not in hd.columns]
    
    if missing_cols:
        print(f"Comparison file missing features: {missing_cols}")
        print("Attempting to rely on 'x.csv' or similar if available, or just re-calculating from raw features if present.")
        # Fallback logic would be needed here, but let's assume for a moment the comparison CSV isn't enough
        # and we need to load the training data to get the features for these specific events.
        print("ABORTING: Comparison CSV lacks features for inference. Please load 'x.csv' or original data.")
        return

    # Assuming we can predict
    # Predict probabilities for each layer
    # The class exposes the internal models: model_block, model_acc, model_finish
    
    # Pre-process if needed (NaN handling etc is done inside the class predict_proba usually, 
    # but we want raw access. Let's try to use the public predict_proba if it were modified, 
    # but here we likely need to manually drive the layers.)
    
    # Let's just create a wrapper to run the layers using the features present
    # We need to respect the feature sets defined in the model
    
    print("Running Layer Predictions via predict_proba_layer...")
    
    try:
        # 1. Block Probability (P_blocked)
        p_blocked = model.predict_proba_layer(hd, layer='block')
        p_unblocked = 1.0 - p_blocked

        # 2. Accuracy Probability (P_on_net)
        p_on_net = model.predict_proba_layer(hd, layer='accuracy')
        
        # 3. Finish Probability (P_goal | on_net)
        p_finish = model.predict_proba_layer(hd, layer='finish')

        # Combine
        p_final = p_unblocked * p_on_net * p_finish
        
        # Analyze
        results = pd.DataFrame({
            'mp_xg': hd['xGoal'],
            'my_xg': p_final,
            'p_unblocked': p_unblocked,
            'p_on_net': p_on_net,
            'p_finish': p_finish,
            'event': hd['event']
        })
        
        print("\n--- Diagnostic: Layer Statistics on High Danger Shots ---")
        print(results.describe())
        
        print("\n--- Identifying the Bottleneck ---")
        print(f"Mean P(Unblocked) (Should be ~1.0): {results['p_unblocked'].mean():.3f}")
        print(f"Mean P(On Net)    (Should be high): {results['p_on_net'].mean():.3f}")
        print(f"Mean P(Finish)    (Should be high): {results['p_finish'].mean():.3f}")
        
        print("\n--- Max Values ---")
        print(f"Max P(Unblocked): {results['p_unblocked'].max():.3f}")
        print(f"Max P(On Net):    {results['p_on_net'].max():.3f}")
        print(f"Max P(Finish):    {results['p_finish'].max():.3f}")

        # Count how often each layer is the "min" (the bottleneck)
        results['bottleneck'] = results[['p_unblocked', 'p_on_net', 'p_finish']].idxmin(axis=1)
        print("\n--- Primary Bottleneck Frequency ---")
        print(results['bottleneck'].value_counts())
        
        # Breakdown one example
        print("\n--- Example Breakdown (Top Discrepancy) ---")
        ex = results.loc[results['p_final'].idxmin()] if 'p_final' in results else results.iloc[0]
        # Recalculate diff to find biggest miss
        results['diff'] = results['mp_xg'] - results['my_xg']
        worst = results.loc[results['diff'].idxmax()]
        print(f"MP xG: {worst['mp_xg']:.3f} | My xG: {worst['my_xg']:.3f}")
        print(f"  > P(Unblocked): {worst['p_unblocked']:.3f}")
        print(f"  > P(On Net):    {worst['p_on_net']:.3f}")
        print(f"  > P(Finish):    {worst['p_finish']:.3f}")
        print(f"  > BOTTLENECK:   {worst['bottleneck']}")

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
