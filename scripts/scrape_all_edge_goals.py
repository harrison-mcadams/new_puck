
import sys
import os
import csv
import json
import time
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck.nhl_api import get_game_feed, get_season
from puck.edge import fetch_tracking_data, transform_coordinates

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

OUTPUT_DIR = os.path.join("data", "edge_goals")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata_v2.csv") # v2 to avoid overwriting existing while testing

METADATA_HEADERS = [
    'season', 'game_id', 'event_id', 'game_date',
    'period', 'time_in_period', 'strength',
    'goal_type', 'secondary_type',
    'scorer_id', 'scorer_name',
    'assist1_id', 'assist1_name',
    'assist2_id', 'assist2_name',
    'goalie_id', 'goalie_name',
    'scoring_team_id', 'home_team_id', 'away_team_id',
    'is_gwg', 'is_empty_net',
    'coord_x', 'coord_y'
]

def get_existing_games():
    """Reads the metadata file to find already processed game_ids."""
    if not os.path.exists(METADATA_FILE):
        return set()
    
    try:
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return set(row['game_id'] for row in reader)
    except Exception as e:
        logging.error(f"Error reading metadata file: {e}")
        return set()

def init_metadata_file():
    """Initializes the metadata CSV with headers if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    if not os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(METADATA_HEADERS)

def extract_metadata(play, game_id, game_date, home_team_id, away_team_id):
    """Extracts comprehensive metadata from a goal play event."""
    # Basic info
    event_id = str(play.get('eventId'))
    period = play.get('periodDescriptor', {}).get('number', 0)
    time_in_period = play.get('timeInPeriod', '00:00')
    strength = play.get('strength', 'EV')
    
    # Details
    details = play.get('details', {})
    goal_type = details.get('shotType', '')
    secondary_type = details.get('secondaryType', '') # Sometimes used for deflection etc
    
    scorer_id = details.get('scoringPlayerId')
    scorer_name = '' # Would need lookup or extraction from parsing description if not in details
    # API v1 usually puts scorer name in a different spot or we need to look it up.
    # Actually, play['details'] usually has 'scoringPlayerId'. Names might be on the roster side.
    # For now, we'll try to grab it if available or leave blank.
    # In 'details', sometimes there is 'scoringPlayerName' (older API) or we rely on roster.
    
    # Assists
    a1_id = details.get('assist1PlayerId', '')
    a1_name = ''
    a2_id = details.get('assist2PlayerId', '')
    a2_name = ''
    
    goalie_id = details.get('goalieInNetId', '')
    goalie_name = ''

    scoring_team_id = details.get('eventOwnerTeamId') # OR check 'team' field at top level
    
    is_gwg = 'gameWinningGoal' in details and details['gameWinningGoal']
    is_empty_net = 'emptyNet' in details and details['emptyNet']
    
    x = details.get('xCoord', '')
    y = details.get('yCoord', '')
    
    return [
        '', # Season filled later
        game_id, event_id, game_date,
        period, time_in_period, strength,
        goal_type, secondary_type,
        scorer_id, scorer_name,
        a1_id, a1_name,
        a2_id, a2_name,
        goalie_id, goalie_name,
        scoring_team_id, home_team_id, away_team_id,
        is_gwg, is_empty_net,
        x, y
    ]

def process_game(game_id, season, writer, limit_goals=None):
    """Process a single game: fetch feed, find goals, scrape edge, save."""
    try:
        feed = get_game_feed(str(game_id))
    except Exception as e:
        logging.error(f"Failed to get feed for game {game_id}: {e}")
        return 0, 0

    game_date = feed.get('gameDate', 'Unknown')
    home_team_id = feed.get('homeTeam', {}).get('id')
    away_team_id = feed.get('awayTeam', {}).get('id')

    # Locate plays
    plays = feed.get('plays', [])
    if not plays:
        plays = feed.get('liveData', {}).get('plays', {}).get('allPlays', [])

    goals_processed = 0
    goals_with_edge = 0

    for play in plays:
        if limit_goals and goals_processed >= limit_goals:
            break

        is_goal = False
        event_id = None
        
        type_desc = play.get('typeDescKey', '')
        play_type = play.get('type', '')
        
        if type_desc == 'goal' or play_type == 'GOAL':
            is_goal = True
            event_id = str(play.get('eventId'))

        if not is_goal or not event_id:
            continue

        goals_processed += 1
        
        # Prepare Metadata Row
        row = extract_metadata(play, game_id, game_date, home_team_id, away_team_id)
        row[0] = season # fill season

        # Edge Fetch
        season_dir = os.path.join(OUTPUT_DIR, str(season))
        os.makedirs(season_dir, exist_ok=True)
        
        json_path = os.path.join(season_dir, f"game_{game_id}_goal_{event_id}_edge.json")
        csv_path = os.path.join(season_dir, f"game_{game_id}_goal_{event_id}_positions.csv")

        edge_data = None
        
        # Check if we already have it locally
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    edge_data = json.load(f)
            except:
                pass # Corrupt? re-fetch
        
        if not edge_data:
            try:
                edge_data = fetch_tracking_data(str(game_id), event_id, str(season))
                if edge_data:
                    with open(json_path, 'w') as f:
                        json.dump(edge_data, f, indent=2)
            except Exception as e:
                logging.warning(f"  Error fetching edge for G{game_id} E{event_id}: {e}")

        if edge_data:
            goals_with_edge += 1
            # Convert to CSV for convenience
            try:
                save_tracking_csv(edge_data, csv_path)
            except Exception as e:
                logging.error(f"  Error saving CSV for G{game_id} E{event_id}: {e}")
        
        # Write metadata regardless of edge availability? 
        # User said "scrape edge data routine", implying we want the edge data.
        # But maybe we want metadata for all goals, and edge where available.
        # Let's save metadata for ALL goals, and maybe add a column 'has_edge'?
        # For now, just save.
        
        # Lock not needed if single threaded, which this is.
        # Open file in append mode each time or pass handle? 
        # Passing handle `writer` is better.
        writer.writerow(row)
        
    return goals_processed, goals_with_edge

def save_tracking_csv(data, filepath):
    """Converts JSON edge data to CSV format."""
    rows = [['frame_idx', 'timestamp', 'entity_type', 'entity_id', 'team_id', 'x', 'y', 'sweater_number']]
    
    if isinstance(data, list):
        for fi, frame in enumerate(data):
            ts = frame.get('timeStamp')
            on_ice = frame.get('onIce', {})
            for k, v in on_ice.items():
                xr, yr = v.get('x'), v.get('y')
                if xr is None or yr is None: continue
                x, y = transform_coordinates(xr, yr)
                
                if k == "1": # Puck
                     rows.append([fi, ts, 'puck', 'puck', '', x, y, ''])
                else:
                     rows.append([fi, ts, 'player', v.get('playerId', k), v.get('teamId', ''), x, y, v.get('sweaterNumber', '')])

    with open(filepath, 'w', newline='') as f:
        csv.writer(f).writerows(rows)

def main():
    parser = argparse.ArgumentParser(description="Scrape NHL Edge data for all goals.")
    parser.add_argument('--limit-games', type=int, help="Limit number of games to process per season (for testing).")
    parser.add_argument('--seasons', nargs='+', default=['20232024', '20242025', '20252026'], help="Seasons to scrape.")
    args = parser.parse_args()

    init_metadata_file()
    existing_games = get_existing_games()
    
    logging.info(f"Starting scraper. Seasons: {args.seasons}")
    logging.info(f"Already processed {len(existing_games)} games.")

    with open(METADATA_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        for season in args.seasons:
            logging.info(f"--- Processing Season {season} ---")
            
            # Fetch full season schedule using 'all' team implies obtaining league schedule?
            # get_season(team='all', ...) uses the weekly paging logic.
            # But get_season usually expects a team abbr? 
            # The definition `get_season(team='all'...)` in nhl_api.py seems to handle league wide.
            # Let's try grabbing all games.
            
            try:
                games = get_season(team='all', season=season)
            except Exception as e:
                logging.error(f"Failed to get schedule for {season}: {e}")
                continue
                
            # Filter regular season only? 
            # Usually strict filtering is good, but playoffs also have data.
            # Let's include Playoffs (03) if available. The get_season default is ['02'].
            # Using defaults for now to match `get_season` behavior unless we change it.
            
            # Identify final games
            final_games = [g for g in games if g.get('gameState') in ['FINAL', 'OFF', 'CRIT']]
            logging.info(f"Found {len(final_games)} final games for {season}.")
            
            count = 0
            for game in final_games:
                game_id = str(game.get('id') or game.get('gamePk'))
                if not game_id: continue
                
                if game_id in existing_games:
                    continue
                
                logging.info(f"Processing Game {game_id} ({count+1}/{len(final_games)})")
                
                gp, ge = process_game(game_id, season, writer)
                
                logging.info(f"  -> {gp} goals, {ge} with Edge data.")
                
                # Flush prevents data loss on interrupt
                f.flush()
                
                count += 1
                if args.limit_games and count >= args.limit_games:
                    logging.info(f"Hit limit of {args.limit_games} games for this season.")
                    break
                    
                time.sleep(0.5) # respectful rate limit

    logging.info("Scraping finished.")

if __name__ == "__main__":
    main()
