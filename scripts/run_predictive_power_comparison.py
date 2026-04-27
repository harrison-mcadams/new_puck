
import subprocess
import os
import re
import pandas as pd

def main():
    # Use the aggregate sweep capability of evaluate_predictive_power.py
    # We compare:
    # 1. Nested GLM (nested)
    # 2. Non-Nested GLM (non_nested)
    # 3. XGBoost Nested (xgboost_nested)
    # 4. XGBoost Non-Nested (xgboost_non_nested)
    # 5. XGBoost Alternate (xgboost_alternate)
    # 6. Actual Goals (actual)
    
    models = "nested,non_nested,xgboost_nested,xgboost_non_nested,xgboost_alternate,actual"
    
    print(f"Running comprehensive predictive power sweep for models: {models}")
    
    cmd = [
        "python", "scripts/evaluate_predictive_power.py",
        "--seasons", "20202021+",
        "--model", models,
        "--filter", "all",
        "--n-boot", "100",
        "--parallel",
        "--n-jobs", "-1",
        "--per-season"
    ]
    
    try:
        # Run and print output live
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        full_output = ""
        for line in process.stdout:
            print(line, end="")
            full_output += line
            
        process.wait()
        
        if process.returncode != 0:
            print(f"\nError: evaluate_predictive_power.py exited with code {process.returncode}")
            return

        print("\n" + "="*80)
        print("PREDICTIVE POWER COMPARISON COMPLETE")
        print("="*80)
        print("The results table above (from evaluate_predictive_power.py) shows the Stability/Predictive metrics.")
        
    except Exception as e:
        print(f"Error running comparison: {e}")

if __name__ == "__main__":
    main()
