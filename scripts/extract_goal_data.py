import argparse
import requests
import json
import csv
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck.nhl_api import get_game_feed, SESSION
from puck.rink import draw_rink

def fetch_tracking_data(game_id, event_id, season):
    """
    Fetches the tracking data for a specific event from the NHL Edge API.
    URL Pattern: https://wsr.nhle.com/sprites/{season}/{gameId}/ev{eventId}.json
    """
    url = f"https://wsr.nhle.com/sprites/{season}/{game_id}/ev{event_id}.json"
    print(f"Fetching tracking data from: {url}")
    
    # Add headers to mimic a browser to avoid 403
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.nhl.com/",
        "Origin": "https://www.nhl.com"
    }

    try:
        resp = SESSION.get(url, headers=headers, timeout=10)
        if resp.status_code == 404:
            print(f"  -> Data not found (404). This goal might not have Edge data.")
            return None
        elif resp.status_code == 403:
            print(f"  -> Forbidden (403). Headers might be insufficient.")
            return None
        
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  -> Error fetching data: {e}")
        return None

def visualize_goal(data, game_id, event_id):
    """
    Visualizes goal tracking data as a Matplotlib animation.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    print("Preparing visualization...")
    
    if not isinstance(data, list) or not data:
        print("No frame data to visualize.")
        return

    # Helper to parse frames
    frames = []
    for frame in data:
        ts = frame.get('timeStamp')
        on_ice = frame.get('onIce', {})
        
        frame_data = {'puck': None, 'players': []}
        
        for key, info in on_ice.items():
            x_raw = info.get('x')
            y_raw = info.get('y')
            if x_raw is None or y_raw is None:
                continue
            
            # Scale to feet (Estimate: 12 units/ft, Origin at 1200, 510)
            x = (x_raw - 1200.0) / 12.0
            y = -(y_raw - 510.0) / 12.0 # Flip Y for correct orientation
                
            if key == "1":
                frame_data['puck'] = (x, y)
            else:
                frame_data['players'].append({
                    'id': key,
                    'sys_id': info.get('playerId'),
                    'team': info.get('teamId'),
                    'x': x,
                    'y': y,
                    'number': info.get('sweaterNumber', '')
                })
        frames.append(frame_data)
        
    print(f"Visualizing {len(frames)} frames...")

    fig, ax = plt.subplots(figsize=(10, 5))
    draw_rink(ax)
    
    # Initialize scatter plots
    # Teams are hard to guess from raw data without roster, but we can color by teamId
    # Usually we get int IDs. Let's create a dynamic colormap
    team_colors = {}
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    
    puck_scatter = ax.scatter([], [], c='black', s=50, zorder=5, label='Puck')
    players_scatter = ax.scatter([], [], c=[], s=100, zorder=4, cmap='bwr', vmin=0, vmax=1)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        puck_scatter.set_offsets(np.empty((0, 2)))
        players_scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return puck_scatter, players_scatter, time_text

    def update(frame_idx):
        frame = frames[frame_idx]
        
        # Puck
        if frame['puck']:
            puck_scatter.set_offsets([frame['puck']])
        else:
            puck_scatter.set_offsets(np.empty((0, 2)))
            
        # Players
        p_xy = []
        p_c = [] # colors (0 or 1 for home/away heuristic)
        
        for p in frame['players']:
            p_xy.append((p['x'], p['y']))
            
            tid = p['team']
            if tid not in team_colors:
                team_colors[tid] = len(team_colors) % 2 # 0 or 1
            p_c.append(team_colors[tid])
            
        if p_xy:
            players_scatter.set_offsets(p_xy)
            players_scatter.set_array(p_c)
        else:
             players_scatter.set_offsets(np.empty((0, 2)))
             
        time_text.set_text(f"Frame {frame_idx}")
        
        return puck_scatter, players_scatter, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=False, interval=50) # 20fps roughly
    
    plt.title(f"Goal Visualization: Game {game_id} Event {event_id}")
    plt.legend(loc='upper right')
    plt.show()

def process_game(game_id, season="20252026", output_dir="data", demo=False):
    """
    Finds all goals in a game and extracts their tracking data.
    """
    print(f"Processing Game ID: {game_id}")
    
    # 1. Get Game Feed to find Goal Events
    try:
        feed = get_game_feed(game_id)
    except Exception as e:
        print(f"Failed to get game feed for {game_id}: {e}")
        return

    # Extract goals (exclude shootout outcomes usually, focusing on regulation/OT goals)
    goals = []
    
    # Handle different API shapes (new/old) - defaulting to new 2023+ shape helpers if available, 
    # but here we'll manually parse the provided feed structure since we know it.
    # The 'plays' are usually under 'plays' list in new API.
    plays = feed.get('plays', [])
    if not plays:
        # Fallback for older API structure if needed
        plays = feed.get('liveData', {}).get('plays', {}).get('allPlays', [])

    for play in plays:
        # Check for Goal event
        # New API: typeDescKey == 'goal'
        # Old API: result.event == 'Goal'
        is_goal = False
        event_id = None
        
        if 'typeDescKey' in play:
            if play['typeDescKey'] == 'goal':
                is_goal = True
                event_id = play.get('eventId')
        elif 'result' in play:
             if play['result'].get('event') == 'Goal':
                is_goal = True
                event_id = play.get('about', {}).get('eventId')

        if is_goal and event_id:
            # Get scorer name for filename context
            goals.append({
                'event_id': event_id,
                'period': play.get('periodDescriptor', {}).get('number', 0),
                'time': play.get('timeInPeriod', '00:00')
            })

    print(f"Found {len(goals)} goals.")
    
    os.makedirs(output_dir, exist_ok=True)

    selected_data = None
    selected_event = None

    for goal in goals:
        event_id = goal['event_id']
        print(f"Processing Goal: Event {event_id} (P{goal['period']} - {goal['time']})")
        
        data = fetch_tracking_data(game_id, event_id, season)
        if not data:
            continue
            
        # Save Raw JSON
        json_filename = os.path.join(output_dir, f"game_{game_id}_goal_{event_id}_edge.json")
        with open(json_filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  -> Saved raw JSON to {json_filename}")
        
        # Process into CSV
        csv_filename = os.path.join(output_dir, f"game_{game_id}_goal_{event_id}_positions.csv")
        csv_rows = []
        
        # Header
        csv_rows.append(['frame_idx', 'timestamp', 'entity_type', 'entity_id', 'team_id', 'x', 'y', 'sweater_number'])
        
        # Frames
        # Data is list of frames
        if isinstance(data, list):
            for i, frame in enumerate(data):
                ts = frame.get('timeStamp')
                on_ice = frame.get('onIce', {})
                
                if not on_ice:
                    continue
                    
                # The keys in onIce are playerIds or "1" for Puck
                for key, info in on_ice.items():
                    if key == "1":
                        # Puck
                        csv_rows.append([
                            i, ts, 'puck', 
                            'puck', # id
                            '', # team
                            info.get('x'), info.get('y'), ''
                        ])
                    else:
                        # Player
                        csv_rows.append([
                            i, ts, 'player', 
                            info.get('playerId', key),
                            info.get('teamId', ''),
                            info.get('x'), info.get('y'),
                            info.get('sweaterNumber', '')
                        ])
        
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"  -> Saved CSV to {csv_filename}")

        if demo:
            selected_data = data
            selected_event = event_id
            # Just take the first valid goal for the demo
            break
            
    if demo and selected_data:
        visualize_goal(selected_data, game_id, selected_event)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract NHL Edge Goal Tracking Data")
    parser.add_argument("--game_id", type=str, default="2025020583", help="Game ID to process")
    parser.add_argument("--season", type=str, default="20252026", help="Season string (e.g., 20252026)")
    parser.add_argument("--output", type=str, default="data/edge_goals", help="Output directory")
    parser.add_argument("--demo", action="store_true", help="Visualize the first valid goal found")
    
    args = parser.parse_args()
    
    process_game(args.game_id, args.season, args.output, args.demo)
