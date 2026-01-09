import sys
import os
import pandas as pd
from pathlib import Path
import glob

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgs, analyze

def show_flyers_recent():
    print("Searching for recent Flyers (PHI) games...")
    
    # 1. Load 2024-2025 or 2025-2026 data
    # Check 20252026 first as that's user context
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    season = '20252026'
    
    path = data_dir / f'{season}.csv'
    if not path.exists():
        # Try finding any season file
        print(f"Season file {path} not found. Searching available...")
        files = glob.glob(f'{data_dir}/*/*.csv')
        if not files:
            print("No data found.")
            return
        # sort by name desc (recent years)
        files.sort(reverse=True)
        path = Path(files[0])
        print(f"Using {path}")
    else:
        print(f"Using {path}")

    # Load data
    # load_data expects a directory where it glob for csvs, OR we can use pd.read_csv directly since we have the path.
    # fit_xgs.load_data is convenient but strict about dirs.
    # Let's use pd.read_csv to be safe with single file this way.
    df = pd.read_csv(path)
    
    # Filter for Flyers
    # "Flyers" -> team_id 4 or abbreviation PHI.
    # We'll look for 'PHI' in 'home_team_abbr' or 'away_team_abbr' maybe?
    # Or just use team_id 4.
    
    # Let's verify team identification.
    # Usually team_id 4 is PHI.
    flyers_id = 4
    
    try:
        df['game_id'] = df['game_id'].astype(int)
    except:
        pass
        
    # Get all game_ids involving PHI
    # We check home_id or away_id if present
    if 'home_id' in df.columns:
        mask_phi = (df['home_id'].astype(str) == str(flyers_id)) | (df['away_id'].astype(str) == str(flyers_id))
    elif 'team_id' in df.columns:
        mask_phi = (df['team_id'].astype(str) == str(flyers_id))
    else:
        print("Could not identify team columns.")
        return

    df_phi = df[mask_phi].copy()
    
    if df_phi.empty:
        print("No Flyers games found in this file.")
        return
        
    # Get most recent game
    recent_game_id = df_phi['game_id'].max()
    print(f"Most recent Flyers game ID: {recent_game_id}")
    
    df_game = df_phi[df_phi['game_id'] == recent_game_id].copy()
    
    # 2. Run Prediction Pipeline
    print("Running xG model on game data...")
    # This calls _predict_xgs -> loads saved model -> uses subcomponents
    df_pred, _, _ = analyze._predict_xgs(df_game)
    
    # 3. Show Results
    # Columns to show
    # Note: analyze.py adds 'xgs', not 'xG'.
    cols = ['game_id', 'period', 'time_in_period', 'event', 'team_id', 'player_id', 'xgs', 'prob_block', 'prob_accuracy', 'prob_finish']
    
    # Handle missing cols gracefully
    show_cols = [c for c in cols if c in df_pred.columns]
    
    # Filter to interesting events (predictions exist)
    # Check for 'xgs' column
    if 'xgs' in df_pred.columns:
        df_show = df_pred.dropna(subset=['xgs'])
    else:
        print("xgs column missing from predictions.")
        return
    
    if df_show.empty:
        print("No predictions made (maybe no shots?).")
    else:
        print(f"Predictions for Game {recent_game_id}:")
        print(df_show[show_cols].head(20).to_string(index=False))
        
        # Show specific sample of blocked shots vs goals
        print("\n--- Sample Blocked Shots ---")
        blocks = df_show[df_show['event'] == 'blocked-shot']
        if not blocks.empty:
            print(blocks[show_cols].head(5).to_string(index=False))
            
        print("\n--- Sample Goals/Shots ---")
        shots = df_show[df_show['event'].isin(['goal', 'shot-on-goal'])]
        if not shots.empty:
            print(shots[show_cols].head(5).to_string(index=False))

    # Enrich with Player Names if possible
    # Try to load cached bios or fetch them
    # We can use simple map if we have it, or just rely on 'puck.nhl_api'
    try:
        from puck import nhl_api
        import sys
        
        # Get unique player IDs
        pids = df_show['player_id'].dropna().unique().astype(int).tolist()
        
        # Fetch bios (this handles caching internally usually)
        # But we need a simpler way if analyze doesn't do it automatically for names
        # analyze.py doesn't automatically add 'player_name'
        
        # Let's try to fetch for this season
        print("Fetching player names...")
        bios = nhl_api.get_season_player_bios(season=season)
        
        # Map
        # bios is a dict of id -> info
        # info has 'lastName', 'firstName'
        
        def get_name(pid):
            if pd.isna(pid): return ''
            try:
                pid_int = int(pid)
                # Check both int and str to be safe
                if pid_int in bios:
                    p = bios[pid_int]
                    return f"{p.get('firstName', '')} {p.get('lastName', '')}"
                elif str(pid_int) in bios:
                    p = bios[str(pid_int)]
                    return f"{p.get('firstName', '')} {p.get('lastName', '')}"
            except:
                pass
            return str(pid)

        df_show['player_name'] = df_show['player_id'].apply(get_name)
        
        # Also map team names if possible
        # We can assume standard team map or just leave team_id/abbr if present
        if 'team_abbr' not in df_show.columns and 'team_id' in df_show.columns:
            # simple hardcoded map for Flyers at least
            df_show['team_name'] = df_show['team_id'].apply(lambda x: 'Flyers' if str(x)=='4' else 'Opponent')
            
    except Exception as e:
        print(f"Could not enrich names: {e}")

    # Save to CSV
    # Ensure context cols are first
    context_cols = ['game_id', 'period', 'time_in_period', 'event', 'team_id', 'team_name', 'player_id', 'player_name', 'x', 'y', 'game_state', 'xgs', 'prob_block', 'prob_accuracy', 'prob_finish']
    
    # Reorder if columns exist
    existing_cols = [c for c in context_cols if c in df_show.columns]
    remainder_cols = [c for c in df_show.columns if c not in existing_cols]
    
    df_final = df_show[existing_cols + remainder_cols]

    out_csv = f'analysis/flyers_recent_predictions_{recent_game_id}.csv'
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_csv, index=False)
    print(f"\nSaved full enriched predictions to {out_csv}")

if __name__ == "__main__":
    show_flyers_recent()
