import os
import sys
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing
from puck import config

def parse_gs_toi(game_id, season='20252026'):
    gid_str = str(game_id)
    if len(gid_str) == 10:
        short_id = gid_str[4:]
    else:
        short_id = gid_str.zfill(6)
        
    url = f"https://www.nhl.com/scores/htmlreports/{season}/GS{short_id}.HTM"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = {
            '5v5': 0,
            'visitor_pp_5v4': 0,
            'home_pp_5v4': 0
        }
        
        def to_seconds(time_str):
            if not time_str or '/' not in time_str: return 0
            ts = time_str.split('/')[-1].strip()
            if not ts or ':' not in ts: return 0
            try:
                if len(ts) > 5: # handle cases like "01:23:45" (rare)
                    ts = ts[-5:]
                mm, ss = ts.split(':')
                return int(mm) * 60 + int(ss)
            except:
                return 0

        # Find all tables
        tables = soup.find_all('table')
        
        # Search for Power Plays
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 2: continue
            
            header_text = rows[0].get_text().upper()
            if '5V4' in header_text and ('PP' in header_text or 'POWER' in soup.get_text().upper()):
                # This could be a PP table. 
                # Let's find which team it belongs to.
                # The PP section has "POWER PLAYS" heading, then a table with 2 cells (Visitor, Home).
                parent_td = table.find_parent('td')
                if parent_td:
                    main_row = parent_td.find_parent('tr')
                    if main_row:
                        main_cells = main_row.find_all('td', recursive=False)
                        if len(main_cells) >= 2:
                            is_home = (parent_td == main_cells[1])
                            
                            header_tds = rows[0].find_all('td')
                            data_tds = rows[1].find_all('td')
                            
                            for i, h_td in enumerate(header_tds):
                                if '5v4' in h_td.get_text():
                                    if len(data_tds) > i:
                                        time_val = to_seconds(data_tds[i].get_text())
                                        if is_home:
                                            results['home_pp_5v4'] = time_val
                                        else:
                                            results['visitor_pp_5v4'] = time_val

        # Search for Even Strength
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 2: continue
            
            header_text = rows[0].get_text().upper()
            if '5V5' in header_text and 'EVEN STRENGTH' in soup.get_text().upper():
                header_tds = rows[0].find_all('td')
                data_tds = rows[1].find_all('td')
                
                for i, h_td in enumerate(header_tds):
                    if '5v5' in h_td.get_text():
                        if len(data_tds) > i:
                            results['5v5'] = to_seconds(data_tds[i].get_text())
                            break

        return results
    except Exception as e:
        print(f"Error parsing game {game_id}: {e}")
        return None

def compare_games(num_games=100):
    # Load season data
    csv_path = 'data/20252026.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    # Filter to pre-season and regular season
    # Pre-season IDs: 202501xxxx, Regular: 202502xxxx
    game_ids = sorted([gid for gid in df['game_id'].unique() if str(gid).startswith('202501') or str(gid).startswith('202502')])
    
    if not game_ids:
        print("No valid game IDs found.")
        return

    # Pick a sample of 100 games
    import numpy as np
    np.random.seed(42)
    sample_gids = np.random.choice(game_ids, min(num_games, len(game_ids)), replace=False)
    sample_gids = sorted(sample_gids.tolist())
    
    comparison = []
    
    print(f"Comparing {len(sample_gids)} games...")
    
    for gid in sample_gids:
        gs_data = parse_gs_toi(gid)
        if not gs_data:
            continue
            
        # Our calculation
        # Note: timing.compute_intervals_for_game defaults to Home perspective
        # which means 5v4 is Home PP and 4v5 is Visitor PP.
        try:
            res_5v5 = timing.compute_intervals_for_game(gid, {'game_state': ['5v5'], 'is_net_empty': [0]}, season='20252026')
            res_5v4 = timing.compute_intervals_for_game(gid, {'game_state': ['5v4'], 'is_net_empty': [0]}, season='20252026')
            res_4v5 = timing.compute_intervals_for_game(gid, {'game_state': ['4v5'], 'is_net_empty': [0]}, season='20252026')
            
            our_5v5 = res_5v5.get('intersection_seconds', 0)
            our_5v4 = res_5v4.get('intersection_seconds', 0)
            our_4v5 = res_4v5.get('intersection_seconds', 0)
            
            gs_5v5 = gs_data['5v5']
            gs_home_pp = gs_data['home_pp_5v4']
            gs_visitor_pp = gs_data['visitor_pp_5v4']
            
            comparison.append({
                'game_id': gid,
                'our_5v5': our_5v5,
                'gs_5v5': gs_5v5,
                'diff_5v5': our_5v5 - gs_5v5,
                'our_5v4': our_5v4,
                'gs_5v4': gs_home_pp,
                'diff_5v4': our_5v4 - gs_home_pp,
                'our_4v5': our_4v5,
                'gs_4v5': gs_visitor_pp,
                'diff_4v5': our_4v5 - gs_visitor_pp
            })
            
            print(f"G {gid} | 5v5: our={our_5v5:.1f} gs={gs_5v5:.1f} diff={our_5v5-gs_5v5:.1f} | 5v4: our={our_5v4:.1f} gs={gs_home_pp:.1f} diff={our_5v4-gs_home_pp:.1f} | 4v5: our={our_4v5:.1f} gs={gs_visitor_pp:.1f} diff={our_4v5-gs_visitor_pp:.1f}")
        except Exception as e:
            print(f"Error computing intervals for {gid}: {e}")
            
        time.sleep(0.1) # Fast but safe
        
    comp_df = pd.DataFrame(comparison)
    os.makedirs('scratch', exist_ok=True)
    comp_df.to_csv('scratch/toi_comparison_results.csv', index=False)
    
    print("\n" + "="*40)
    print("TOI COMPARISON SUMMARY (Our Logic vs NHL GS Report)")
    print("="*40)
    stats = comp_df[['diff_5v5', 'diff_5v4', 'diff_4v5']].describe()
    print(stats)
    
    # Calculate percentage of games with 0 difference
    perfect_5v5 = (comp_df['diff_5v5'] == 0).sum() / len(comp_df) * 100
    perfect_pp = (comp_df['diff_5v4'] == 0).sum() / len(comp_df) * 100
    perfect_pk = (comp_df['diff_4v5'] == 0).sum() / len(comp_df) * 100
    
    print(f"\nPerfect Match Rate (0s diff):")
    print(f"  5v5: {perfect_5v5:.1f}%")
    print(f"  5v4: {perfect_pp:.1f}%")
    print(f"  4v5: {perfect_pk:.1f}%")
    
    # Median absolute error
    mae_5v5 = comp_df['diff_5v5'].abs().median()
    mae_pp = comp_df['diff_5v4'].abs().median()
    mae_pk = comp_df['diff_4v5'].abs().median()
    
    print(f"\nMedian Absolute Error:")
    print(f"  5v5: {mae_5v5:.1f}s")
    print(f"  5v4: {mae_pp:.1f}s")
    print(f"  4v5: {mae_pk:.1f}s")

if __name__ == "__main__":
    compare_games(100)
