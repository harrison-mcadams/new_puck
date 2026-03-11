
import joblib
import pandas as pd
import numpy as np

def inspect_model():
    model_path = "analysis/xgs/xg_model_nested_tensor.joblib"
    model = joblib.load(model_path)
    
    print(f"Model: {type(model)}")
    print(f"Features: {model.features}")
    
    # Check if finish model is fitted
    finish_model = model.model_finish
    if finish_model:
        preprocessor = finish_model.named_steps['preprocessor']
        cat_transformer = preprocessor.named_transformers_['cat']
        ohe = cat_transformer.named_steps['ohe']
        
        feature_names = ohe.get_feature_names_out()
        print("\nCategorical Feature Names (Finish Model):")
        # Find all relative_game_state features
    for layer_name in ['model_block', 'model_acc', 'model_finish']:
        layer = getattr(model, layer_name)
        if layer:
            print(f"\n--- {layer_name.upper()} ---")
            preprocessor = layer.named_steps['preprocessor']
            clf = layer.named_steps['clf']
            
            try:
                 all_feature_names = preprocessor.get_feature_names_out()
                 coefs = clf.coef_[0]
                 for i, f in enumerate(all_feature_names):
                      if 'relative_game_state' in f and ('5v4' in f or '4v5' in f or '5v5' in f):
                           print(f"  [{i}] {f:<40}: {coefs[i]:.4f}")
            except Exception as e:
                 print(f"  Could not get features: {e}")

if __name__ == "__main__":
    inspect_model()
