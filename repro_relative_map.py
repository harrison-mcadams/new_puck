import analyze
import os
import matplotlib.pyplot as plt

# Ensure static directory exists
os.makedirs('static', exist_ok=True)

# Run season analysis for a single team (e.g., ANA)
# We assume 20252026 data exists or can be computed.
# If not, we might fail. Let's try to find a valid team/season.
# We'll just try '20252026' and 'ANA' as per the plan.

if __name__ == "__main__":
    print("Running season_analysis for ANA...")
    try:
        # Use a small number of events or mock if needed, but let's try real execution first
        # This assumes league baseline can be computed or loaded.
        res = analyze.season_analysis(season='20252026', teams=['ANA'], baseline_mode='load')
        
        if 'ANA' in res['teams']:
            print("ANA processed successfully.")
            out_path = res['teams']['ANA'].get('relative_map_path')
            if out_path and os.path.exists(out_path):
                print(f"Output file exists: {out_path}")
            else:
                print(f"Output file missing: {out_path}")
                
            # Check summary table
            print("Summary Table Head:")
            print(res['summary_table'].head())
        else:
            print("ANA not found in results.")
            print(res)

    except Exception as e:
        print(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
