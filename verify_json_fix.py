
import json
import numpy as np
import os

def run_test():
    print("--- Testing JSON Serialization Fix ---")
    
    # Mock data with numpy types
    league_xg = np.float64(123.456)
    league_seconds = np.float64(789.012)
    
    print(f"Original types: league_xg={type(league_xg)}, league_seconds={type(league_seconds)}")
    
    # Simulate the fix: casting to float
    results = {
        'season': '20252026',
        'league': {
            'xg_total': float(league_xg),
            'seconds': float(league_seconds)
        }
    }
    
    print(f"Result types: xg_total={type(results['league']['xg_total'])}, seconds={type(results['league']['seconds'])}")
    
    # Try saving
    out_path = 'test_baseline.json'
    try:
        with open(out_path, 'w') as f:
            json.dump(results, f)
        print("SUCCESS: JSON saved successfully.")
        
        # Verify load
        with open(out_path, 'r') as f:
            loaded = json.load(f)
        print(f"Loaded: {loaded}")
        
    except Exception as e:
        print(f"FAILURE: JSON dump failed: {e}")
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)

if __name__ == '__main__':
    run_test()
