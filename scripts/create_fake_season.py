#!/usr/bin/env python3
"""
scripts/create_fake_season.py

Generates a controllable fake season to robustly test the analysis pipeline.
Features:
1. Creates `data/fake2025.csv` with simulated events across 5v5, 5v4, 4v5.
2. Creates `data/fake2025/shifts/*.pkl` with simulated shifts.
3. Defines two teams with distinct spatial behaviors:
   - Team RIGHT (ID 100): Only shoots from Y < -10.
   - Team LEFT (ID 200): Only shoots from Y > 10.
4. Verification:
   - Runs `scripts/daily.py`.
   - Checks if generated heatmaps respect spatial rules.
   - Checks if xG totals match known inputs.

Usage:
    python3 scripts/create_fake_season.py
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import shutil
import subprocess
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import config

SEASON = 'fake2025'
DATA_DIR = os.path.join('data', SEASON)
SHIFTS_DIR = os.path.join(DATA_DIR, 'shifts')

# Expected Totals Accumulator
EXPECTED = {
    '100': {'5v5': 0.0, '5v4': 0.0, '4v5': 0.0},
    '200': {'5v5': 0.0, '5v4': 0.0, '4v5': 0.0}
}

def setup_dirs():
    if os.path.exists(DATA_DIR):
        print(f"Cleaning existing {DATA_DIR}...")
        shutil.rmtree(DATA_DIR)
    os.makedirs(SHIFTS_DIR, exist_ok=True)

def create_fake_data(n_games=4):
    print(f"Generating {n_games} fake games...")
    
    events = []
    
    TEAM_R_ID = 100
    TEAM_L_ID = 200
    TEAM_R_ABB = 'RGT'
    TEAM_L_ABB = 'LFT'
    
    # Update expected keys
    global EXPECTED
    EXPECTED = {
        str(TEAM_R_ID): {'5v5': 0.0, '5v4': 0.0, '4v5': 0.0},
        str(TEAM_L_ID): {'5v5': 0.0, '5v4': 0.0, '4v5': 0.0}
    }
    
    games = []
    
    for i in range(n_games):
        gid = 2025000001 + i
        games.append(gid)
        
        # Determine Home/Away (Alternate)
        if i % 2 == 0:
            home_id, home_abb = TEAM_R_ID, TEAM_R_ABB
            away_id, away_abb = TEAM_L_ID, TEAM_L_ABB
        else:
            home_id, home_abb = TEAM_L_ID, TEAM_L_ABB
            away_id, away_abb = TEAM_R_ID, TEAM_R_ABB
            
        home_def_side = 'left' 
        
        # Simulate Events
        # We want to simulate different states.
        # Let's say:
        # 0-1200s: 5v5
        # 1200-2400s: 5v4 (Home Powerplay)
        # 2400-3600s: 4v5 (Home Penalty Kill / Away Powerplay)
        
        # Events generation
        for team_type in ['home', 'away']:
            tid = home_id if team_type == 'home' else away_id
            tabb = home_abb if team_type == 'home' else away_abb
            is_r_team = (tid == TEAM_R_ID)
            
            # 5v5 Shots (Period 1)
            for k in range(5):
                sec = np.random.uniform(0, 1200)
                add_event(events, gid, sec, '5v5', tid, home_id, away_id, home_abb, away_abb, home_def_side, is_r_team)
            
            # 5v4 Shots (Period 2) - Home has 5, Away has 4.
            # If tid matches Home, they are in 5v4 state.
            # If tid matches Away, they are in 4v5 state (SH).
            state = '5v4' if team_type == 'home' else '4v5'
            for k in range(5):
                sec = np.random.uniform(1200, 2400)
                add_event(events, gid, sec, state, tid, home_id, away_id, home_abb, away_abb, home_def_side, is_r_team)
                
            # 4v5 Shots (Period 3) - Home has 4, Away has 5.
            # If tid matches Home -> 4v5
            # If tid matches Away -> 5v4
            state = '4v5' if team_type == 'home' else '5v4'
            for k in range(5):
                sec = np.random.uniform(2400, 3600)
                add_event(events, gid, sec, state, tid, home_id, away_id, home_abb, away_abb, home_def_side, is_r_team)
        
        # Create Shifts
        # Period 1: 5v5 (5 skaters each + G)
        # Period 2: 5v4 (Home 5+G, Away 4+G)
        # Period 3: 4v5 (Home 4+G, Away 5+G)
        
        shift_rows = []
        
        # Goalies - All Game
        add_shift(shift_rows, gid, home_id, 30, 'G', 0, 3600)
        add_shift(shift_rows, gid, away_id, 30, 'G', 0, 3600)
        
        # Home Skaters
        # P1 (5v5): 5 skaters
        for k in range(1, 6): add_shift(shift_rows, gid, home_id, k, 'S', 0, 1200)
        # P2 (5v4): 5 skaters (PP)
        for k in range(1, 6): add_shift(shift_rows, gid, home_id, k, 'S', 1200, 2400)
        # P3 (4v5): 4 skaters (PK)
        for k in range(1, 5): add_shift(shift_rows, gid, home_id, k, 'S', 2400, 3600)
            
        # Away Skaters
        # P1 (5v5): 5 skaters
        for k in range(1, 6): add_shift(shift_rows, gid, away_id, k, 'S', 0, 1200)
        # P2 (5v4 - PK): 4 skaters
        for k in range(1, 5): add_shift(shift_rows, gid, away_id, k, 'S', 1200, 2400)
        # P3 (4v5 - PP): 5 skaters
        for k in range(1, 6): add_shift(shift_rows, gid, away_id, k, 'S', 2400, 3600)

        df_shifts = pd.DataFrame(shift_rows)
        pkl_path = os.path.join(SHIFTS_DIR, f'shifts_{gid}.pkl')
        df_shifts.to_pickle(pkl_path)
    
    # Save CSV
    df_events = pd.DataFrame(events)
    # Ensure is_net_empty = 0
    df_events['is_net_empty'] = 0
    
    csv_path = os.path.join('data', f"{SEASON}.csv")
    df_events.to_csv(csv_path, index=False)
    print(f"Saved fake season CSV to {csv_path}")
    
    return TEAM_R_ABB, TEAM_L_ABB

def add_event(events, gid, sec, state, tid, home_id, away_id, home_abb, away_abb, home_def, is_r_team):
    period = 1 if sec < 1200 else (2 if sec < 2400 else 3)
    p_time = sec % 1200
    p_time_str = f"{int(p_time//60):02d}:{int(p_time%60):02d}"
    
    # Spatial Logic
    # Team R (is_r_team): Y < -10
    # Team L (not is_r_team): Y > 10
    y = np.random.uniform(-40, -10) if is_r_team else np.random.uniform(10, 40)
    
    # Attack Side
    # Home defends Left -> Attacks Right (X > 0)
    # Away attacks Left (X < 0)
    is_home = (tid == home_id)
    if (is_home and home_def == 'left') or (not is_home and home_def == 'right'):
        x = np.random.uniform(30, 80)
    else:
        x = np.random.uniform(-80, -30)
    
    xg_val = 0.05
    
    events.append({
        'game_id': gid,
        'period': period,
        'period_time': p_time_str,
        'total_time_elapsed_seconds': sec,
        'game_state': state, # This is the NAIVE state (dataframe column)
        'is_net_empty': 0,
        'team_id': tid,
        'home_id': home_id,
        'away_id': away_id,
        'home_abb': home_abb,
        'away_abb': away_abb,
        'home_team_defending_side': home_def,
        'event': 'shot-on-goal',
        'x': x,
        'y': y,
        'player_id': tid * 10 + 1,
        'player_name': f"P{tid}",
        'shot_type': 'Wrist Shot',
        'xg': xg_val,
        'xgs': xg_val,
        'period_type': 'REGULAR'
    })
    
    # Update Expectation
    if str(tid) in EXPECTED:
        EXPECTED[str(tid)][state] += xg_val

def add_shift(rows, gid, tid, number, type_code, start, end):
    pid = tid * 10 + number
    rows.append({
        'game_id': gid,
        'team_id': tid,
        'player_id': pid,
        'player_name': f"Player_{pid}",
        'start_total_seconds': float(start),
        'end_total_seconds': float(end),
        'period': 1,
        'detail_code': 0,
        'duration': float(end - start),
        'type_code': 0 # doesn't matter much for internal logic if we use explicit classification
    })

def run_pipeline():
    print("Running Daily Pipeline (Cache + Stats) on fake season...")
    from puck import config
    try:
        partials_dir = os.path.join(config.get_cache_dir(SEASON), 'partials')
        os.makedirs(partials_dir, exist_ok=True)
    except: pass

    # Do NOT pass --force because daily.py deletes the season CSV if force=True!
    cmd = [sys.executable, 'scripts/daily.py', '--season', SEASON]
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Pipeline failed!")
        print("STDOUT:", res.stdout[-2000:])
        print("STDERR:", res.stderr[-2000:])
        return False
    # print("Pipeline Output:", res.stdout[-500:]) 
    return True

def verify_totals():
    print("\nVerifying xG Totals against Analysis Output...")
    # Load 5v5, 5v4, 4v5 summaries
    
    failures = 0
    
    for cond in ['5v5', '5v4', '4v5']:
        summary_path = os.path.join('analysis', 'league', SEASON, cond, 'team_summary.json')
        if not os.path.exists(summary_path):
            print(f"FAIL: Summary for {cond} missing.")
            failures += 1
            continue
            
        with open(summary_path, 'r') as f:
            data = json.load(f) # List of dicts
            
        # Convert to dict by team_id (we need to map abb to id, or just check both)
        # The fake season uses IDs 100, 200. Names Team Right, Team Left.
        # run_league_stats outputs team names in 'team' field? Or abbreviation?
        # Usually abbreviation.
        
        for row in data:
            team_lbl = row.get('team')
            # Map back to ID
            tid = None
            if team_lbl == 'RGT': tid = '100'
            elif team_lbl == 'LFT': tid = '200'
            
            if tid:
                actual_xg = row.get('team_xgs', 0.0) # wait, summary usually has stats dict?
                # run_league_stats output list of dicts: {'team': 'PHI', 'team_xgs': ..., ...}
                # Let's check keys.
                # If keys are just rates (per60), we might need to convert back?
                # run_league_stats saves: team, team_goals, team_xgs, etc. YES raw totals are there.
                
                expected = EXPECTED[tid][cond]
                
                # Check with tolerance
                if abs(actual_xg - expected) > 0.01:
                    print(f"FAIL: {team_lbl} {cond} xG mismatch. Expected {expected:.2f}, Got {actual_xg:.2f}")
                    failures += 1
                else:
                    print(f"PASS: {team_lbl} {cond} xG matches ({expected:.2f})")
    
    return failures == 0

def verify_spatial_logic(team_r, team_l):
    # Same as before...
    part_dir = os.path.join(config.get_cache_dir(SEASON), 'partials')
    files = [f for f in os.listdir(part_dir) if f.endswith('5v5.npz')]
    
    print(f"\nVerifying spatial patterns in {len(files)} partials...")
    violations = 0
    for f in files:
        with np.load(os.path.join(part_dir, f), allow_pickle=True) as data:
            if 'team_100_grid_team' in data:
                grid = data['team_100_grid_team']
                h_mass = np.sum(grid[:, 43:]) 
                l_mass = np.sum(grid[:, :43])
                if h_mass > l_mass:
                     print(f"FAIL: Team R (Right) has more mass in Top Half!")
                     violations += 1
            if 'team_200_grid_team' in data:
                grid = data['team_200_grid_team']
                h_mass = np.sum(grid[:, 43:]) 
                l_mass = np.sum(grid[:, :43])
                if l_mass > h_mass:
                     print(f"FAIL: Team L (Left) has more mass in Bottom Half!")
                     violations += 1
    
    if violations == 0:
        print("PASS: Spatial verification successful.")
    else:
        print(f"FAIL: Spatial verification failed with {violations} violations.")
    return violations == 0

def generate_verification_plots():
    print("\nGenerating verification plots...")
    try:
        import matplotlib.pyplot as plt
        from puck.rink import draw_rink
        
        part_dir = os.path.join(config.get_cache_dir(SEASON), 'partials')
        files = [f for f in os.listdir(part_dir) if f.endswith('5v5.npz')]
        
        # Accumulate grids
        # Resolutions are usually 201x86? No, check array shape.
        # Analyze uses: range [[-100, 100], [-42.5, 42.5]]
        # We'll just init with None and add.
        
        grid_r = None
        grid_l = None
        
        for f in files:
            with np.load(os.path.join(part_dir, f), allow_pickle=True) as data:
                if 'team_100_grid_team' in data:
                    g = data['team_100_grid_team']
                    if grid_r is None: grid_r = np.zeros_like(g)
                    grid_r += g
                if 'team_200_grid_team' in data:
                    g = data['team_200_grid_team']
                    if grid_l is None: grid_l = np.zeros_like(g)
                    grid_l += g
        
        if grid_r is None or grid_l is None:
            print("No grids found to plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot Team Right (100)
        ax = axes[0]
        draw_rink(ax)
        # Grid is typically (X, Y) or (Y, X)?
        # analyze.compute_xg_heatmap_from_df returns gx, gy, heatmap.
        # heatmap is shape (Nx, Ny) usually.
        # extent corresponds to [minX, maxX, minY, maxY].
        # Resolution 1.0 -> 200 units X, 85 units Y.
        
        # Actually usually it's transposed for imshow?
        # imshow expects (Rows, Cols) -> (Y, X).
        # if grid is (Nx, Ny), we need to transpose.
        # Let's assume standard behavior: Transpose for imshow.
        
        extent = [-100, 100, -42.5, 42.5]
        
        ax.imshow(grid_r.T, extent=extent, origin='lower', cmap='Reds', alpha=0.7)
        ax.set_title(f"Team Right (100) - Expected: Bottom Half")
        
        # Plot Team Left (200)
        ax = axes[1]
        draw_rink(ax)
        ax.imshow(grid_l.T, extent=extent, origin='lower', cmap='Blues', alpha=0.7)
        ax.set_title(f"Team Left (200) - Expected: Top Half")
        
        out_path = os.path.join('analysis', 'fake_season_verification.png')
        plt.savefig(out_path, dpi=150)
        print(f"Saved verification plot to {out_path}")
        
    except Exception as e:
        print(f"Failed to generate plots: {e}")


def main():
    setup_dirs()
    create_fake_data()
    if run_pipeline():
        vt = verify_totals()
        vs = verify_spatial_logic('RGT', 'LFT')
        generate_verification_plots()
        
        if vt and vs:
            print("\nSUCCESS: Fake Season Verification Passed!")
            sys.exit(0)
        else:
            print("\nFAILURE: Fake Season Verification Failed.")
            sys.exit(1)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
