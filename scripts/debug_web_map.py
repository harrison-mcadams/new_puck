
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import puck.config as puck_config
from puck import analyze

# Simulate app.py's ANALYSIS_DIR
# app.py: ANALYSIS_DIR = os.path.abspath("analysis")
# debug_web_map.py is in scripts/, so cwd might be different when running.
# But let's assume we run from project root.
APP_ANALYSIS_DIR = os.path.abspath("analysis")

def debug_xgs_map():
    print(f"DEBUG: puck_config.ANALYSIS_DIR = {puck_config.ANALYSIS_DIR}")
    print(f"DEBUG: app.py ANALYSIS_DIR = {APP_ANALYSIS_DIR}")

    print(f"DEBUG: puck_config.DATA_DIR = {puck_config.DATA_DIR}")
    
    # Simulate params from app.py
    # ret = analyze.xgs_map(
    #         game_id=game_val,
    #         condition=condition,
    #         out_path=out_path,
    #         show=False,
    #         return_heatmaps=False,
    #         events_to_plot=['shot-on-goal', 'goal', 'xgs'],
    #         return_filtered_df=True,
    #         force_refresh=True
    #     )
    
    game_id = 2025010077 # app.py passes int if it's all digits
    # The first line has 2025020703
    
    condition = {'game_state': ['5v5']} # Simulate 5v5 replot

    out_path = os.path.join(APP_ANALYSIS_DIR, 'debug_test_map.png')
    
    print(f"DEBUG: calling xgs_map with game_id={game_id}, out_path={out_path}")
    
    try:
        ret = analyze.xgs_map(
            game_id=game_id,
            condition=condition,
            out_path=out_path,
            show=False,
            return_heatmaps=True,
            events_to_plot=['shot-on-goal', 'goal', 'xgs'],
            return_filtered_df=True,
            force_refresh=True,
            season='20252026' # Make sure we use the right season for file lookup
        )
        print("DEBUG: xgs_map returned successfully.")
        
        # Check heatmaps
        import numpy as np
        heatmaps = ret[1]
        print(f"DEBUG: Heatmaps keys: {heatmaps.keys() if heatmaps else 'None'}")
        if heatmaps:
            for k, v in heatmaps.items():
                if v is not None:
                    print(f"DEBUG: Heatmap '{k}' max: {np.nanmax(v)}, sum: {np.nansum(v)}")
                else:
                    print(f"DEBUG: Heatmap '{k}' is None")

        # Check Summary Stats
        stats = ret[3]
        print(f"DEBUG: Summary Stats: {stats}")

        if os.path.exists(out_path):
             print(f"DEBUG: Output file created at {out_path}")
        else:
             print(f"DEBUG: Output file NOT created at {out_path}")
             
        # Check DF for xgs
        df = ret[2]
        if df is not None:
             print(f"DEBUG: Returned DF shape: {df.shape}")
             if 'xgs' in df.columns:
                 print(f"DEBUG: 'xgs' column present. NaNs: {df['xgs'].isna().sum()}, Valid: {df['xgs'].notna().sum()}")
                 
                 # Check shots specifically
                 shots = df[df['event'].isin(['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot'])]
                 print(f"DEBUG: Shot events count: {len(shots)}")
                 if not shots.empty:
                     print(f"DEBUG: Shot xG sample: {shots['xgs'].head().tolist()}")
                     print(f"DEBUG: Max xG: {shots['xgs'].max()}")
                     print(f"DEBUG: Mean xG: {shots['xgs'].mean()}")
                 else:
                     print("DEBUG: No shot events found in this game sample.")

                 print(f"DEBUG: All events xgs sample: {df['xgs'].dropna().head().tolist()}")
             else:
                 print("DEBUG: 'xgs' column MISSING from DF")
        else:
             print("DEBUG: Returned DF is None")

    except Exception as e:
        print(f"DEBUG: xgs_map crashed with: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_xgs_map()
