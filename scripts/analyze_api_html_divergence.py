
import pandas as pd
import numpy as np
import re
import os
import sys
import logging
from bs4 import BeautifulSoup

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import nhl_api, parse, config

def mmss_to_seconds(mmss: str) -> int:
    try:
        if ':' not in mmss:
            return int(float(mmss))
        mm, ss = mmss.split(':')
        return int(mm) * 60 + int(ss)
    except Exception:
        return 0

def parse_html_pbp(html_text):
    if not html_text:
        return []
    
    soup = BeautifulSoup(html_text, 'html.parser')
    # Find all tables and look for the one containing 'Event' and 'Description'
    # Instead of one table, let's find all rows on the page that look like PBP
    all_rows = soup.find_all('tr')
    events = []
    current_period = 1
    
    for row in all_rows:
        tds = row.find_all('td')
        row_txt = row.get_text().upper()
        
        # Period detection
        if "PERIOD" in row_txt and len(tds) < 5:
            m = re.search(r'(\d+)\s*(?:ST|ND|RD|TH)?\s*PERIOD', row_txt)
            if m:
                current_period = int(m.group(1))
            elif "OT PERIOD" in row_txt:
                current_period = 4
            continue

        if len(tds) < 8:
            continue
            
        first_col = tds[0].get_text(strip=True)
        if not first_col.isdigit():
            if "PERIOD" in row_txt:
                m = re.search(r'(\d+)\s*(?:ST|ND|RD|TH)?\s*PERIOD', row_txt)
                if m:
                    current_period = int(m.group(1))
            continue
            
        event_num = int(first_col)
        period = current_period
        
        # Time columns: tds[3] is Elapsed <br> Remaining
        # tds[3].get_text(" ", strip=True) gives "0:00 20:00"
        raw_time = tds[3].get_text(" ", strip=True)
        time_parts = raw_time.split()
        if not time_parts:
            continue
            
        elapsed_str = time_parts[0]
        elapsed_sec = mmss_to_seconds(elapsed_str)
        
        # Event code at index 4, Description at index 5
        event_code = tds[4].get_text(strip=True).upper()
        description = tds[5].get_text(strip=True)
        
        # Extract distance
        dist_match = re.search(r'(\d+)\s*ft\.', description)
        distance = float(dist_match.group(1)) if dist_match else None
        
        events.append({
            'html_event_id': event_num,
            'period': period,
            'period_seconds': elapsed_sec,
            'event_code': event_code,
            'html_description': description,
            'html_distance': distance
        })
        
    return events

def analyze_divergence(game_id):
    # 1. Get API Data
    api_feed = nhl_api.get_game_feed(game_id)
    if not api_feed:
        return None
        
    api_df = parse._game(api_feed)
    api_df = parse._elaborate(api_df)
    
    # 2. Get HTML Data
    html_text = nhl_api.get_pbp_from_nhl_html(game_id)
    html_events = parse_html_pbp(html_text)
    html_df = pd.DataFrame(html_events)
    
    if html_df.empty:
        return None
        
    # 3. Match Events
    # Filter API for shots
    shot_types = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    api_shots = api_df[api_df['event'].isin(shot_types)].copy()
    
    # Filter HTML for shots (SHOT, MISS, GOAL, BLOCK)
    html_shot_codes = ['SHOT', 'MISS', 'GOAL', 'BLOCK']
    html_shots = html_df[html_df['event_code'].isin(html_shot_codes)].copy()
    
    # Determine time column
    time_cols = [c for c in api_shots.columns if 'time' in c.lower()]
    t_col = 'time_elapsed_in_period_s' if 'time_elapsed_in_period_s' in api_shots.columns else (time_cols[0] if time_cols else None)
    
    if not t_col:
        return None
        
    results = []
    for _, api_row in api_shots.iterrows():
        # Match by period and time (within 5 seconds)
        matches = html_shots[
            (html_shots['period'] == api_row['period']) &
            (np.abs(html_shots['period_seconds'] - api_row[t_col]) <= 5)
        ]
        
        if not matches.empty:
            # Take the best match (closest time)
            best_match = matches.assign(
                time_diff=np.abs(matches['period_seconds'] - api_row[t_col])
            ).sort_values('time_diff').iloc[0]
            
            if best_match['html_distance'] is not None:
                results.append({
                    'event': api_row['event'],
                    'api_dist': api_row['distance'],
                    'html_dist': best_match['html_distance'],
                    'diff': api_row['distance'] - best_match['html_distance'],
                    'player': api_row.get('player_name', 'Unknown')
                })
                
    return pd.DataFrame(results)

def main():
    # Let's check a few games from the 2025 season
    games = list(range(2025020107, 2025020127))
    
    # Suppress verbose logging
    logging.getLogger('puck').setLevel(logging.ERROR)
    
    all_results = []
    for gid in games:
        res = analyze_divergence(gid)
        if res is not None:
            all_results.append(res)
            
    if not all_results:
        print("No matches found in any game.")
        return
        
    df = pd.concat(all_results)
    if df.empty:
        print("No paired events found across all games.")
        return
    
    print("\n=== Global Divergence Summary ===")
    print(f"Total Matches: {len(df)}")
    print(f"Mean Abs Error: {df['diff'].abs().mean():.2f} ft")
    print(f"Median Abs Error: {df['diff'].abs().median():.2f} ft")
    
    # Divergence buckets
    print("\nDivergence Buckets:")
    print(f"  < 2ft:   {(df['diff'].abs() < 2).sum()} ({ (df['diff'].abs() < 2).mean()*100:.1f}%)")
    print(f"  2-10ft:  {((df['diff'].abs() >= 2) & (df['diff'].abs() < 10)).sum()} ({ ((df['diff'].abs() >= 2) & (df['diff'].abs() < 10)).mean()*100:.1f}%)")
    print(f"  10-50ft: {((df['diff'].abs() >= 10) & (df['diff'].abs() < 50)).sum()} ({ ((df['diff'].abs() >= 10) & (df['diff'].abs() < 50)).mean()*100:.1f}%)")
    print(f"  50-100ft:{((df['diff'].abs() >= 50) & (df['diff'].abs() < 100)).sum()} ({ ((df['diff'].abs() >= 50) & (df['diff'].abs() < 100)).mean()*100:.1f}%)")
    print(f"  > 100ft: {(df['diff'].abs() >= 100).sum()} ({ (df['diff'].abs() >= 100).mean()*100:.1f}%)")
    
    if not df[df['diff'].abs() >= 20].empty:
        print("\n--- Significant Divergences (> 20ft) ---")
        print(df[df['diff'].abs() >= 20].sort_values('diff', ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
