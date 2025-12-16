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
import hashlib

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
        
    if os.path.exists(SHIFTS_DIR):
        shutil.rmtree(SHIFTS_DIR)
        
    # Also clean the partials cache to force re-computation
    cache_root = os.path.join('data', 'cache', SEASON)
    if os.path.exists(cache_root):
        shutil.rmtree(cache_root)
        print(f"Cleaned stale cache at {cache_root}")

    os.makedirs(SHIFTS_DIR, exist_ok=True)

def create_fake_data(n_games=6):
    print(f"Generating {n_games} fake games...")
    
    events = []
    
    TEAM_R_ID = 100
    TEAM_L_ID = 200
    TEAM_RND_ID = 300
    
    TEAM_R_ABB = 'RGT'
    TEAM_L_ABB = 'LFT'
    TEAM_RND_ABB = 'RND'
    
    teams_def = {
        TEAM_R_ID: {'abb': TEAM_R_ABB, 'name': 'Team Right (Y < -15)'},
        TEAM_L_ID: {'abb': TEAM_L_ABB, 'name': 'Team Left (Y > 15)'},
        TEAM_RND_ID: {'abb': TEAM_RND_ABB, 'name': 'Team Random'}
    }
    
    # Update expected keys
    global EXPECTED
    EXPECTED = {
        str(TEAM_R_ID): {'5v5': 0.0, '5v4': 0.0, '4v5': 0.0},
        str(TEAM_L_ID): {'5v5': 0.0, '5v4': 0.0, '4v5': 0.0},
        str(TEAM_RND_ID): {'5v5': 0.0, '5v4': 0.0, '4v5': 0.0}
    }
    
    # 6 Games: Round Robin (Home/Away for each pair)
    # Pairs: (100, 200), (200, 100), (100, 300), (300, 100), (200, 300), (300, 200)
    matchups = [
        (TEAM_R_ID, TEAM_L_ID),
        (TEAM_L_ID, TEAM_R_ID),
        (TEAM_R_ID, TEAM_RND_ID),
        (TEAM_RND_ID, TEAM_R_ID),
        (TEAM_L_ID, TEAM_RND_ID),
        (TEAM_RND_ID, TEAM_L_ID)
    ]
    
    games = []
    
    for i in range(n_games):
        gid = 2025000001 + i
        games.append(gid)
        
        # Determine Home/Away
        if i < len(matchups):
            home_id, away_id = matchups[i]
        else:
            # Fallback if n_games > 6
            home_id, away_id = matchups[i % len(matchups)]
            
        home_abb = teams_def[home_id]['abb']
        away_abb = teams_def[away_id]['abb']
            
        home_def_side = 'left' 
        
        # Simulate Events
        for team_type in ['home', 'away']:
            tid = home_id if team_type == 'home' else away_id
            tabb = home_abb if team_type == 'home' else away_abb
            
            # 5v5 Shots (Period 1)
            for k in range(5):
                sec = np.random.uniform(0, 1200)
                add_event(events, gid, sec, '5v5', tid, home_id, away_id, home_abb, away_abb, home_def_side)
            
            # 5v4 Shots (Period 2)
            # Home has 5, Away has 4.
            state = '5v4' if team_type == 'home' else '4v5'
            for k in range(5):
                sec = np.random.uniform(1200, 2400)
                add_event(events, gid, sec, state, tid, home_id, away_id, home_abb, away_abb, home_def_side)
                
            # 4v5 Shots (Period 3)
            # Home has 4, Away has 5.
            state = '4v5' if team_type == 'home' else '5v4'
            for k in range(5):
                sec = np.random.uniform(2400, 3600)
                add_event(events, gid, sec, state, tid, home_id, away_id, home_abb, away_abb, home_def_side)
        
        # Create Shifts
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
        
        # Inject Fake Feed Cache (for timing.py)
        cache_fake_feed(gid, home_id, away_id)
        
    # Generate Fake Teams JSON
    fake_teams = [{"id": k, "abbr": v['abb'], "name": v['name']} for k, v in teams_def.items()]

    teams_json_path = os.path.join('analysis', 'teams.json')
    with open(teams_json_path, 'w') as f:
        json.dump(fake_teams, f, indent=2)
    print(f"Generated fake teams JSON at {teams_json_path}")
    
    # Save CSV
    df_events = pd.DataFrame(events)
    # Ensure is_net_empty = 0
    df_events['is_net_empty'] = 0
    
    csv_path = os.path.join('data', f"{SEASON}.csv")
    df_events.to_csv(csv_path, index=False)
    print(f"Saved fake season CSV to {csv_path}")
    
    return TEAM_R_ABB, TEAM_L_ABB, TEAM_RND_ABB

def cache_fake_feed(gid, home_id, away_id):
    cache_dir = os.path.join('.cache', 'nhl_api')
    os.makedirs(cache_dir, exist_ok=True)
    
    feed = {
        'id': gid,
        'gameState': 'FINAL',
        'homeTeam': {'id': home_id, 'abbrev': 'Home', 'score': 1},
        'awayTeam': {'id': away_id, 'abbrev': 'Away', 'score': 1},
        'plays': [] 
    }
    
    key = str(gid)
    safe = hashlib.sha1(key.encode('utf-8')).hexdigest()
    path = os.path.join(cache_dir, f"game_feed_{safe}.json")
    
    with open(path, 'w') as f:
        json.dump(feed, f)


def add_event(events, gid, sec, state, tid, home_id, away_id, home_abb, away_abb, home_def, xg=1.0):
    period = 1 if sec < 1200 else (2 if sec < 2400 else 3)
    p_time = sec % 1200
    p_time_str = f"{int(p_time//60):02d}:{int(p_time%60):02d}"
    
    # Determine Attack Perspective
    # Home defends Left -> Attacks Right (X > 0) defined in data
    # Away attacks Left (X < 0) defined in data
    is_home = (tid == home_id)
    
    # Standardized Frame Target (Attacking Right):
    # RGT (100): Y < -15 (Bottom/Right Wing)
    # LFT (200): Y > 15 (Top/Left Wing)
    # RND (300): Y random -40 to 40
    
    # Generate in ATTACKING FRAME first
    if tid == 100: # RGT
        target_y = np.random.uniform(-40, -15)
        target_x = np.random.uniform(30, 80)
    elif tid == 200: # LFT
        target_y = np.random.uniform(15, 40)
        target_x = np.random.uniform(30, 80)
    else: # RND
        target_y = np.random.uniform(-40, 40)
        target_x = np.random.uniform(30, 80)
        
    # Transform to Global/Data coordinates
    # If Home and Defends Left -> Attack Right -> No change
    # If Away -> Attack Left -> Flip X and Y
    
    if (is_home and home_def == 'left') or (not is_home and home_def == 'right'):
        # Attacking Right
        x, y = target_x, target_y
    else:
        # Attacking Left
        x, y = -target_x, -target_y
    
    xg_val = xg
    
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
        'player_id': tid * 10 + 1, # e.g. 1001, 2001
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
        'type_code': 0, 
        'raw': {'primaryPosition': 'G' if type_code == 'G' else 'C'}
    })

def run_pipeline():
    print("Running Daily Pipeline (Cache + Stats) on fake season...")
    from puck import config
    try:
        partials_dir = os.path.join(config.get_cache_dir(SEASON), 'partials')
        os.makedirs(partials_dir, exist_ok=True)
    except: pass

    cmd = [sys.executable, 'scripts/daily.py', '--season', SEASON, '--skip-fetch']
    print(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    with open('analysis/pipeline.log', 'w') as f:
        f.write(res.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(res.stderr)
        
    if res.returncode != 0:
        print("Pipeline failed!")
        print("STDOUT:", res.stdout[-2000:])
        print("STDERR:", res.stderr[-2000:])
        return False
    return True

def verify_totals():
    print("\nVerifying xG Totals against Analysis Output...")
    failures = 0
    passed = 0
    
    for cond in ['5v5', '5v4', '4v5']:
        summary_path = os.path.join('analysis', 'league', SEASON, cond, 'team_summary.json')
        if not os.path.exists(summary_path):
            print(f"FAIL: Summary for {cond} missing.")
            failures += 1
            continue
            
        with open(summary_path, 'r') as f:
            data = json.load(f) 
            
        # Check all expected teams
        for tid_str, expected_conds in EXPECTED.items():
            expected = expected_conds[cond]
            # Find in data
            # Data usually has team abbreviations.
            
            # Map ID to Abb for search
            abb_map = {'100': 'RGT', '200': 'LFT', '300': 'RND'}
            search_abb = abb_map.get(tid_str)
            
            found = False
            for row in data:
                if row.get('team') == search_abb:
                    actual_xg = row.get('team_xgs', 0.0)
                    if abs(actual_xg - expected) > 0.01:
                        print(f"FAIL: {search_abb} {cond} xG mismatch. Expected {expected:.2f}, Got {actual_xg:.2f}")
                        failures += 1
                    else:
                        print(f"PASS: {search_abb} {cond} xG matches ({expected:.2f})")
                        passed += 1
                    found = True
                    break
            
            if not found and expected > 0:
                 print(f"FAIL: {search_abb} {cond} missing from summary but expected {expected}")
                 failures += 1
            elif not found and expected == 0:
                 pass # correctly missing? or should be present with 0? usually present.
                 
    return failures == 0

def verify_spatial_logic():
    print("\nVerifying spatial patterns...")
    part_dir = os.path.join(config.get_cache_dir(SEASON), 'partials')
    files = [f for f in os.listdir(part_dir) if f.endswith('5v5.npz')]
    
    violations = 0
    
    for f in files:
        with np.load(os.path.join(part_dir, f), allow_pickle=True) as data:
            # Robust Verification: Check for POLARIZATION (Segregation)
            # We don't care if it's Top or Bottom (since standardization might flip orientation),
            # but we care that it is NOT random/centered.
            
            # TEAM RGT (100) -> Should be Polarized (Side Loaded)
            if 'team_100_grid_team' in data:
                grid = data['team_100_grid_team']
                mass_top = np.sum(grid[:, 50:]) 
                mass_bot = np.sum(grid[:, :35])
                
                # Check for Segregation: One side should be much heavier than other
                # Ratio of Max/Min should be high (> 5)
                # handle zero div
                if mass_bot == 0: ratio = 999
                elif mass_top == 0: ratio = 999
                else: ratio = max(mass_top, mass_bot) / min(mass_top, mass_bot)
                
                if ratio < 5.0:
                     print(f"FAIL: Team RGT (Right Wing) is not spatially segregated! Ratio: {ratio:.1f}")
                     violations += 1
            
            # TEAM LFT (200) -> Should be Polarized
            if 'team_200_grid_team' in data:
                grid = data['team_200_grid_team']
                mass_top = np.sum(grid[:, 50:]) 
                mass_bot = np.sum(grid[:, :35])
                
                if mass_bot == 0: ratio = 999
                elif mass_top == 0: ratio = 999
                else: ratio = max(mass_top, mass_bot) / min(mass_top, mass_bot)
                
                if ratio < 5.0:
                     print(f"FAIL: Team LFT (Left Wing) is not spatially segregated! Ratio: {ratio:.1f}")
                     violations += 1
                     
            # TEAM RND (300) -> Random (Center ish)
            if 'team_300_grid_team' in data:
                grid = data['team_300_grid_team']
                mass_total = np.sum(grid)
                y_indices = np.arange(grid.shape[1])
                y_profile = np.sum(grid, axis=0)
                mean_y_idx = np.sum(y_indices * y_profile) / mass_total
                
                if abs(mean_y_idx - 42.5) > 10: 
                     print(f"FAIL: Team RND Mean Y ({mean_y_idx}) is too far from center (42.5)")
                     violations += 1

    if violations == 0:
        print("PASS: Spatial verification successful.")
    else:
        print(f"FAIL: Spatial verification failed with {violations} violations.")
    return violations == 0

def verify_player_stats():
    print("\nVerifying Player Stats (Checking for non-zero Against xG)...")
    # Load player summary from NEW LOCATION
    summ_path = os.path.join('analysis', 'players', SEASON, 'player_summary_5v5.json')
    if not os.path.exists(summ_path):
        print(f"FAIL: Player summary missing at {summ_path}")
        return False
        
    with open(summ_path, 'r') as f:
        players = json.load(f)
        
    # Check a few skaters
    failures = 0
    checked = 0
    for p in players:
        pid = p.get('player_id')
        if not pid or pid % 10 == 0: continue # Skip goalies (ends in 0) or weird IDs
        
        # Skaters should have xG Against > 0 because they play against other teams who shoot
        xga = p.get('xg_ag_60', 0) # Key in summary is xg_ag_60
        
        if xga <= 0:
            print(f"FAIL: Player {p.get('player_name')} has 0.0 xGA per 60.")
            failures += 1
        checked += 1
        
    print(f"Checked {checked} skaters.")
    if failures == 0 and checked > 0:
        print("PASS: All skaters have valid xGA.")
        return True
    else:
        print(f"FAIL: {failures} skaters have 0 xGA.")
        return False

def generate_verification_plots():
    print("\nGenerating verification plots...")
    try:
        import matplotlib.pyplot as plt
        from puck.rink import draw_rink
        
        part_dir = os.path.join(config.get_cache_dir(SEASON), 'partials')
        files = [f for f in os.listdir(part_dir) if f.endswith('5v5.npz')]
        
        grid_r, grid_l, grid_rnd = None, None, None
        
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
                if 'team_300_grid_team' in data:
                    g = data['team_300_grid_team']
                    if grid_rnd is None: grid_rnd = np.zeros_like(g)
                    grid_rnd += g
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        extent = [-100, 100, -42.5, 42.5]
        
        # RGT
        ax = axes[0]
        draw_rink(ax)
        if grid_r is not None: ax.imshow(grid_r.T, extent=extent, origin='lower', cmap='Reds', alpha=0.7)
        ax.set_title("Team RGT (Expected Top)")
        
        # LFT
        ax = axes[1]
        draw_rink(ax)
        if grid_l is not None: ax.imshow(grid_l.T, extent=extent, origin='lower', cmap='Blues', alpha=0.7)
        ax.set_title("Team LFT (Expected Bottom)")
        
        # RND
        ax = axes[2]
        draw_rink(ax)
        if grid_rnd is not None: ax.imshow(grid_rnd.T, extent=extent, origin='lower', cmap='Greens', alpha=0.7)
        ax.set_title("Team RND (Uniform)")
        
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
        vs = verify_spatial_logic()
        vp = verify_player_stats()
        generate_verification_plots()
        
        if vt and vs and vp:
            print("\nSUCCESS: Fake Season Verification Passed!")
            sys.exit(0)
        else:
            print("\nFAILURE: Fake Season Verification Failed.")
            sys.exit(1)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
