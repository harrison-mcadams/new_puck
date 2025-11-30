
import analyze
import os
import shutil

def run_repro():
    out_dir = 'static/test_players_repro'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("Running player analysis for PHI (limited to 1 player for speed if possible, or just let it run for a bit)")
    # We can pass a specific player_id if we know one, or just let it find one.
    # Let's try to find a player ID from the season data first to be fast.
    import timing
    df = timing.load_season_df('20252026')
    if df is None or df.empty:
        print("No season data found, cannot run repro.")
        return

    # Pick a player from PHI or any team if PHI not found
    phi_games = df[(df['home_abb'] == 'PHI') | (df['away_abb'] == 'PHI')]
    
    if phi_games.empty:
        print("No PHI games found. Trying any game.")
        phi_games = df
    
    if phi_games.empty:
        print("No games found at all.")
        return

    # Get a player ID
    # We need a player who has played.
    # Let's just pick the first player_id from a valid event.
    
    sample_pid = None
    sample_team = 'PHI'
    
    # Filter for events with a player_id
    players_df = phi_games.dropna(subset=['player_id'])
    
    if not players_df.empty:
        # Try to find a PHI player specifically if possible
        if 'home_abb' in players_df.columns:
            phi_home = players_df[players_df['home_abb'] == 'PHI']
            if not phi_home.empty:
                # Pick a player from home team (PHI)
                # We need to ensure the player belongs to PHI (home_id)
                # player_id is just the actor.
                # We can check team_id.
                home_id = phi_home.iloc[0]['home_id']
                phi_events = phi_home[phi_home['team_id'] == home_id]
                if not phi_events.empty:
                    sample_pid = int(phi_events.iloc[0]['player_id'])
                    sample_team = 'PHI'
        
        if not sample_pid:
            # Fallback to any player
            sample_pid = int(players_df.iloc[0]['player_id'])
            # Try to determine team
            row = players_df.iloc[0]
            if row['team_id'] == row['home_id']:
                sample_team = row['home_abb']
            else:
                sample_team = row['away_abb']
    
    if sample_pid:
        print(f"Running analysis for all players on {sample_team} to generate scatter plot")
        # Get all player IDs for the team
        team_players = []
        if sample_team == 'PHI':
            # We already have phi_games
            pass
        else:
            # Filter for the sample team
            phi_games = df[(df['home_abb'] == sample_team) | (df['away_abb'] == sample_team)]
            
        # Extract unique player IDs for this team
        # We need to be careful to get players who actually played for this team
        # A simple heuristic: players in events where team_id matches the team's id
        # Get team ID first
        team_id = None
        for _, row in phi_games.iterrows():
            if row['home_abb'] == sample_team:
                team_id = row['home_id']
                break
            elif row['away_abb'] == sample_team:
                team_id = row['away_id']
                break
        
        if team_id:
            team_events = phi_games[phi_games['team_id'] == team_id]
            team_players = team_events['player_id'].dropna().unique().astype(int).tolist()
            print(f"Found {len(team_players)} players for {sample_team}")
            
            # Limit to top 5 players by event count to speed up if needed, or just run all?
            # Running all might take a minute. Let's try top 10.
            # Count events per player
            p_counts = team_events['player_id'].value_counts()
            top_players = p_counts.head(10).index.astype(int).tolist()
            print(f"Analyzing top 10 players: {top_players}")
            
            analyze.players(season='20252026', player_ids=top_players, out_dir=out_dir, team=sample_team)
        else:
            print(f"Could not determine team ID for {sample_team}")
    else:
        print("Could not find a player ID.")

if __name__ == '__main__':
    run_repro()
