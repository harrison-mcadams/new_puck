
import os
import sys
import glob
import pandas as pd
import requests
import warnings
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.join(os.getcwd()))

# Import existing logic (by importing as module or mostly re-implementing orchestration)
# To avoid refactoring the entire identify script right now, let's subprocess it or import its logic if possible.
# But subprocess allows us to capture stdout cleanly and keep state isolated.
# However, importing functions is cleaner.

# Let's import the core function from identify_shot_attempts if we can slightly mod it to return dict.
# Or we just replicate the orchestration here. It is safer to replicate the orchestration loop here
# so we can control the batch process better, but reuse the 'identification' logic.

# Actually, let's try to run the existing scripts as subprocesses first. 
# It's robust and reuses the exact verified code.

import subprocess

DATA_DIR = "data/edge_goals"
ANALYSIS_DIR = "analysis/blocked_shots"
SUMMARY_FILE = os.path.join(ANALYSIS_DIR, "blocked_shots_summary_batch.csv")

def parse_pbp_time(time_str):
    """Parses MM:SS into seconds."""
    if not time_str or ':' not in time_str: return 0
    m, s = time_str.split(':')
    return int(m) * 60 + int(s)

def get_block_info_for_goal(game_id, goal_id):
    """
    Queries NHL API to find the Block Event ID preceding the goal.
    Returns: (block_id, delta_t) or (None, None)
    """
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200: return None, None
        data = resp.json()
    except Exception as e:
        print(f"  [API Error] {e}")
        return None, None

    plays = data.get('plays', [])
    
    # scan for goal
    goal_idx = -1
    goal_time = 0
    goal_period = 0
    for i, p in enumerate(plays):
        if str(p.get('eventId')) == str(goal_id):
            goal_idx = i
            goal_time = parse_pbp_time(p.get('timeInPeriod'))
            goal_period = p.get('periodDescriptor', {}).get('number')
            break
            
    if goal_idx == -1: return None, None

    # Look backwards for Block
    for i in range(goal_idx - 1, max(-1, goal_idx - 50), -1):
        p = plays[i]
        evt = p.get('typeDescKey', '')
        eid = p.get('eventId')
        period = p.get('periodDescriptor', {}).get('number')
        
        if evt == 'blocked-shot' and period == goal_period:
            bt = parse_pbp_time(p.get('timeInPeriod'))
            delta = goal_time - bt
            # Delta should be positive (Block before Goal) 
            # and within 14 seconds (typical edge clip window)
            if 0 < delta <= 14:
                return eid, delta

    return None, None

def main():
    if not os.path.exists(ANALYSIS_DIR):
        os.makedirs(ANALYSIS_DIR)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from existing summary file')
    args = parser.parse_args()

    # Find all position files across all season subdirectories
    # Structure: data/edge_goals/20232024/*.csv, data/edge_goals/20242025/*.csv
    pattern = os.path.join(DATA_DIR, "**", "game_*_goal_*_positions.csv")
    files = glob.glob(pattern, recursive=True)
    print(f"Found {len(files)} position files across all seasons.")

    # Sort for consistency
    files.sort()

    summary_records = []
    processed_keys = set()

    if args.resume and os.path.exists(SUMMARY_FILE):
        print(f"Resuming from {SUMMARY_FILE}...")
        try:
            df_existing = pd.read_csv(SUMMARY_FILE)
            for _, row in df_existing.iterrows():
                processed_keys.add(f"{row['game_id']}_{row['goal_id']}")
            print(f"  Found {len(processed_keys)} already processed goals.")
            
            # Load existing records to keep appending correctly? 
            # Or just append mode? Using pandas to_csv(mode='a') is cleaner but header management is tricky.
            # Simpler: We are appending to summary_records list in loop and writing full file.
            # If resume, we should Load ALL existing into summary_records so they are preserved on overwrite.
            summary_records = df_existing.to_dict('records')
            
        except Exception as e:
            print(f"  [WARN] Failed to read summary file for resume: {e}")

    for fpath in files:
        # Parse Filename: game_2024020151_goal_293_positions.csv
        basename = os.path.basename(fpath)

        parts = basename.split('_')
        if len(parts) < 4: continue
        
        game_id = parts[1]
        goal_id = parts[3]
        
        if f"{game_id}_{goal_id}" in processed_keys:
            continue
            
        print(f"\n=== Processing Game {game_id} Goal {goal_id} ===")
        
        # 1. Get Block ID & Timing
        block_id, delta_t = get_block_info_for_goal(game_id, goal_id)
        if not block_id:
            # Note: This skip is now much more meaningful (no sequence-aligned block found)
            print(f"  [SKIP] No PBP block found within 14s.")
            continue
            
        print(f"  Block ID: {block_id} (Delta: {delta_t}s)")
        
        # 2. Run Identification Script
        # scripts/identify_shot_attempts.py <game> <goal> <block>
        # We capture the output CSV via pandas later.
        
        cmd_id = [sys.executable, "scripts/identify_shot_attempts.py", str(game_id), str(goal_id), str(block_id)]
        try:
            subprocess.run(cmd_id, check=True, capture_output=True, text=True)
            print("  [OK] Identification complete.")
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Identification script failed: {e}")
            # print(e.stderr)
            continue
            
        # 3. Validation / Check Result
        # Load the generated CSV: analysis/blocked_shots/candidate_vectors_{game}_{goal}_v2.csv
        csv_path = os.path.join(ANALYSIS_DIR, f"candidate_vectors_{game_id}_{goal_id}_v2.csv")
        if not os.path.exists(csv_path):
            print("  [WARN] No output CSV generated.")
            continue
            
        df_res = pd.read_csv(csv_path)
        if df_res.empty:
            print("  [WARN] Results CSV is empty.")
            continue
            
        # Get Best Candidate (First row, assuming logic sorts by score)
        best = df_res.iloc[0]
        
        # 4. Visualization (GIF + PNG)
        # scripts/visualize_candidate_vectors.py <game> <goal>
        cmd_viz = [sys.executable, "scripts/visualize_candidate_vectors.py", str(game_id), str(goal_id)]
        try:
            print("  Generating visualizations...")
            subprocess.run(cmd_viz, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Viz failed: {e}")
            
        # 5. Compile Summary Data
        record = {
            'game_id': game_id,
            'goal_id': goal_id,
            'block_id': block_id,
            'delta_t': delta_t,
            'best_frame': best.get('frame_idx'),
            'score': best.get('score'),
            'speed_fps': best.get('speed'),
            'distance_to_shooter': best.get('dist'),
            'distance_to_blocker': best.get('blocker_dist'),
            'net_dev_deg': best.get('dev_deg'),
            'x': best.get('x'),
            'y': best.get('y'),
            'csv_path': csv_path,
            'viz_gif': os.path.join(ANALYSIS_DIR, f"candidate_vectors_{game_id}_{goal_id}.gif"),
            'viz_png': os.path.join(ANALYSIS_DIR, f"candidate_vectors_{game_id}_{goal_id}_best.png")
        }
        summary_records.append(record)
        
        # Incremental Save
        pd.DataFrame(summary_records).to_csv(SUMMARY_FILE, index=False)
        
    print(f"\nBatch processing complete. Summary saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
