"""
Bulk Scraper for Flyers Goal Tracking Data.

This script iterates through specified seasons, finds all goals scored by the Philadelphia Flyers,
and downloads the corresponding NHL Edge tracking data. It maintains a metadata index for easy access.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck.nhl_api import get_game_feed
from puck.edge import fetch_tracking_data

# Constants
TARGET_TEAM = 'PHI'
OUTPUT_DIR = os.path.join("data", "edge_goals")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")

def get_season_games(season):
    """Fetch all games for a specific team in a season."""
    print(f"Fetching schedule for {season}...")
    try:
        # We can use the puck.nhl_api tool or endpoints directly.
        # Since get_season_schedule wraps the schedule endpoint, let's use it.
        # However, it returns a filtered list usually.
        # Let's use the explicit team schedule endpoint for reliability if needed,
        # but for now I'll assume get_season_schedule works or I'll iterate game IDs.
        # Actually, let's rely on the robust `puck.nhl_api` if possible, but
        # a simple generic schedule fetch is safer for a specific team.
        
        # Using the standard NHL schedule endpoint for the team
        import requests
        url = f"https://api-web.nhle.com/v1/club-schedule-season/{TARGET_TEAM}/{season}"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        return data.get('games', [])
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return []

def scrape_season(season):
    """Scrape all Flyers goals for a given season."""
    print(f"\n--- Scraping Season {season} ---")
    
    games = get_season_games(season)
    print(f"Found {len(games)} games for {TARGET_TEAM}.")
    
    goals_found = 0
    goals_downloaded = 0
    
    # Ensure season dir exists
    season_dir = os.path.join(OUTPUT_DIR, season)
    os.makedirs(season_dir, exist_ok=True)
    
    # Load or Create Metadata
    metadata_exists = os.path.exists(METADATA_FILE)
    mode = 'a' if metadata_exists else 'w'
    
    with open(METADATA_FILE, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not metadata_exists:
            writer.writerow(['season', 'game_id', 'event_id', 'date', 'period', 'time', 'scorer_id', 'scorer_name', 'strength'])
        
        for game in games:
            game_id = str(game['id'])
            game_date = game['gameDate']
            
            # Optimization: Check if we already processed this game?
            # For now, we'll process all to be safe, but skip existing files implies we skip fetch.
            
            try:
                feed = get_game_feed(game_id)
            except Exception as e:
                print(f"  Failed to get feed for {game_id}: {e}")
                continue
                
            # Parse goals
            # Helper to find goals for PHI
            plays = feed.get('plays', [])
            if not plays:
                 plays = feed.get('liveData', {}).get('plays', {}).get('allPlays', [])
            
            for play in plays:
                is_goal = False
                scoring_team_id = None
                scoring_team_abbrev = None
                
                # Check goal type
                if 'typeDescKey' in play:
                    if play['typeDescKey'] == 'goal':
                        is_goal = True
                        scoring_team_id = play.get('details', {}).get('eventOwnerTeamId')
                        # Need to resolve ID to Abbrev or just check ID if we knew PHI's ID (which is 4)
                        # Use loose check if possible, or assume 4.
                        # Actually most feeds have 'scoringTeam' or 'details.eventOwnerTeamId'
                        # Let's rely on extracting the abbrev if available or ID.
                        # PHI ID = 4.
                        if scoring_team_id == 4:
                            scoring_team_abbrev = 'PHI'
                            
                elif 'result' in play:
                    if play['result'].get('event') == 'Goal':
                        is_goal = True
                        scoring_team_abbrev = play.get('team', {}).get('triCode')
                
                # If it's a PHI goal
                if is_goal and (scoring_team_abbrev == 'PHI' or scoring_team_id == 4):
                    goals_found += 1
                    
                    # Extract Metadata
                    event_id = play.get('eventId') or play.get('about', {}).get('eventId')
                    period = play.get('periodDescriptor', {}).get('number', 0)
                    time_in_period = play.get('timeInPeriod', '00:00')
                    
                    # Scorer
                    scorer_id = None
                    scorer_name = "Unknown"
                    strength = "EV"
                    
                    # Parsing detailed metadata varies by API version slightly
                    if 'details' in play: # New API
                        scorer_id = play['details'].get('scoringPlayerId')
                        strength = play.get('strength', 'EV') # might need mapping
                    elif 'players' in play: # Old API
                        for p in play['players']:
                            if p['playerType'] == 'Scorer':
                                scorer_id = p['player']['id']
                                scorer_name = p['player']['fullName']
                        strength = play.get('result', {}).get('strength', {}).get('code', 'EV')

                    # File Path
                    json_filename = os.path.join(season, f"game_{game_id}_goal_{event_id}_edge.json")
                    full_json_path = os.path.join(OUTPUT_DIR, json_filename)
                    
                    # Check if already downloaded
                    if os.path.exists(full_json_path):
                        # print(f"  Skipping existing: {json_filename}")
                        continue
                        
                    # Fetch
                    print(f"  Downloading Goal: Game {game_id} Event {event_id} ({game_date})")
                    data = fetch_tracking_data(game_id, event_id, season)
                    
                    if data:
                        # Save
                        with open(full_json_path, 'w') as jf:
                            json.dump(data, jf, indent=2)
                        
                        # Log to CSV
                        writer.writerow([
                            season, game_id, event_id, game_date, 
                            period, time_in_period, scorer_id, scorer_name, strength
                        ])
                        f.flush() # Ensure write
                        goals_downloaded += 1
                        
                        # Be nice to the API
                        time.sleep(0.5)
                    else:
                        print(f"    -> No Edge data available.")

    print(f"Season {season} Complete. Found {goals_found} goals, Downloaded {goals_downloaded} new.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk Scrape Flyers Goal Tracking Data")
    parser.add_argument("--seasons", nargs='+', default=["20242025", "20252026"], help="Seasons to scrape")
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for s in args.seasons:
        scrape_season(s)
