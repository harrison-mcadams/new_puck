
import os
import sys
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import mixed_effects
from puck import mixed_effects_viz

def main():
    model_path = "analysis/xgs/joint_mixed_effects.joblib"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        mixed = joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Model loaded. Feature Set: {mixed.feature_set}")
    print(f"Use Tensor Splines: {mixed.use_tensor_splines}")
    print(f"Models: {mixed.models_.keys()}")
    
    print(f"Main tensor_transformer_: {getattr(mixed, 'tensor_transformer_', 'Missing')}")
            
    print("Generating Spatial Maps and League Scatter...")
    try:
        mixed_effects_viz.generate_spatial_grids(mixed, output_dir="analysis/xgs/mixed_effects/viz")
        print("Visualization complete.")
    except Exception as e:
        print(f"Failed to generate viz: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
