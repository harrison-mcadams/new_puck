import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck.nhl_api import get_game_feed, get_season
from puck.edge import fetch_tracking_data, transform_coordinates

# Constants
OUTPUT_DIR = os.path.join("data", "edge_goals")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")

def scrape_league(season="20252026", limit=None):
    print(f"\n--- Scraping League Goals for Season {season} ---")
    
    # 1. Fetch Global Schedule
    games = get_season(team='all', season=season, game_types=['02'])
    print(f"Found {len(games)} regular season games in global schedule.")
    
    # Ensure season dir exists
    season_dir = os.path.join(OUTPUT_DIR, season)
    os.makedirs(season_dir, exist_ok=True)
    
    # Load existing metadata to skip duplicates
    existing_events = set()
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['season'] == season:
                    existing_events.add(f"{row['game_id']}_{row['event_id']}")

    print(f"Already have {len(existing_events)} goal events in metadata.")

    # 2. Process Games
    goals_downloaded = 0
    
    # Sort games by date or ID
    games.sort(key=lambda x: str(x.get('id', '')))
    
    metadata_f = open(METADATA_FILE, 'a', newline='', encoding='utf-8')
    writer = csv.writer(metadata_f)
    if os.path.getsize(METADATA_FILE) == 0:
        writer.writerow(['season', 'game_id', 'event_id', 'date', 'period', 'time', 'scorer_id', 'scorer_name', 'strength'])
    
    try:
        for i, game in enumerate(games):
            if limit and goals_downloaded >= limit:
                print(f"Reached limit of {limit} goals. Stopping.")
                break
                
            game_id = str(game['id'])
            game_date = game.get('gameDate', 'Unknown')
            
            # Substantial progress info
            if i % 50 == 0:
                print(f"Progress: Game {i}/{len(games)} ({game_id}) - Downloads: {goals_downloaded}")

            # Optimization: We can't easily know if a game has goals without the feed, 
            # but we can check if we've already downloaded ANY goals for this game 
            # by checking the filenames in the season dir.
            # However, new goals might have been scored in the same game ID if we only 
            # partially scraped before. Let's just fetch the feed.
            
            try:
                feed = get_game_feed(game_id)
            except Exception as e:
                # print(f"  Failed to get feed for {game_id}: {e}")
                continue
                
            plays = feed.get('plays', [])
            if not plays:
                 plays = feed.get('liveData', {}).get('plays', {}).get('allPlays', [])
            
            for play in plays:
                is_goal = False
                event_id = None
                
                if play.get('typeDescKey') == 'goal':
                    is_goal = True
                    event_id = str(play.get('eventId'))
                elif play.get('result', {}).get('event') == 'Goal':
                    is_goal = True
                    event_id = str(play.get('about', {}).get('eventId'))
                
                if not is_goal or not event_id:
                    continue
                    
                # Skip if already have it
                if f"{game_id}_{event_id}" in existing_events:
                    continue
                
                # Filter for 5v5 (situationCode check)
                # situationCode: [awayGoalie, awaySkaters, homeSkaters, homeGoalie]
                sc = play.get('situationCode', '0000')
                if not (len(sc) == 4 and sc[1] == '5' and sc[2] == '5'):
                    # Skip non-5v5 (PP, PK, Empty Net, etc.)
                    continue

                # File Path
                json_filename = os.path.join(season, f"game_{game_id}_goal_{event_id}_edge.json")
                full_json_path = os.path.join(OUTPUT_DIR, json_filename)
                csv_filename = os.path.join(season_dir, f"game_{game_id}_goal_{event_id}_positions.csv")
                
                data = None
                if os.path.exists(full_json_path):
                    if os.path.exists(csv_filename):
                        continue
                    # Missing CSV! Load JSON and generate it
                    with open(full_json_path, 'r') as jf:
                        data = json.load(jf)
                else:
                    # Fetch New
                    data = fetch_tracking_data(game_id, event_id, season)
                    if data:
                        with open(full_json_path, 'w') as jf:
                            json.dump(data, jf, indent=2)

                if data:
                    # PROCESS TO CSV
                    csv_rows = [['frame_idx', 'timestamp', 'entity_type', 'entity_id', 'team_id', 'x', 'y', 'sweater_number']]
                    
                    if isinstance(data, list):
                        for fi, frame in enumerate(data):
                            ts = frame.get('timeStamp')
                            on_ice = frame.get('onIce', {})
                            for k, v in on_ice.items():
                                xr, yr = v.get('x'), v.get('y')
                                if xr is None or yr is None: continue
                                x, y = transform_coordinates(xr, yr)
                                if k == "1":
                                    csv_rows.append([fi, ts, 'puck', 'puck', '', x, y, ''])
                                else:
                                    csv_rows.append([fi, ts, 'player', v.get('playerId', k), v.get('teamId', ''), x, y, v.get('sweaterNumber', '')])
                    
                    with open(csv_filename, 'w', newline='') as cf:
                        cw = csv.writer(cf)
                        cw.writerows(csv_rows)

                    # Only log to metadata if it's strictly NEW
                    if f"{game_id}_{event_id}" not in existing_events:
                        period = play.get('periodDescriptor', {}).get('number', 0)
                        time_in_period = play.get('timeInPeriod', '00:00')
                        scorer_id = play.get('details', {}).get('scoringPlayerId')
                        strength = play.get('strength', 'EV')
                        
                        writer.writerow([
                            season, game_id, event_id, game_date, 
                            period, time_in_period, scorer_id, "Unknown", strength
                        ])
                        metadata_f.flush()
                        goals_downloaded += 1
                    
                    # Respect API - slower to avoid 403s
                    time.sleep(0.5)
                    
                    if limit and goals_downloaded >= limit:
                        break
                else:
                    # Mark as attempted/failed? For now just skip.
                    pass
    finally:
        metadata_f.close()

    print(f"League scrape complete. Downloaded {goals_downloaded} new 5v5 goals.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Edge Goal Tracking for the entire League")
    parser.add_argument("--season", type=str, default="20252026", help="Season to scrape")
    parser.add_argument("--limit", type=int, default=None, help="Max goals to download this run")
    args = parser.parse_args()
    
    scrape_league(args.season, args.limit)
