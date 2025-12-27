
import pandas as pd
import sys

def verify_score():
    path = 'data/20252026/20252026_df.csv'
    print(f"Loading {path}...")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("File not found. Wait for re-parse to finish.")
        sys.exit(1)
        
    required = ['score_diff', 'home_score', 'away_score', 'total_time_elapsed_s', 'period_number']
    missing = [c for c in required if c not in df.columns]
    
    if missing:
        print(f"FAILED: Missing columns: {missing}")
        sys.exit(1)
        
    print("SUCCESS: All columns present.")
    
    # Check Goal Logic
    goals = df[df['event'] == 'goal']
    if goals.empty:
        print("Warning: No goals found in this chunk?")
        sys.exit(0)
        
    print(f"Checking {len(goals)} goals for pre-event scoring logic...")
    
    # We want to see if the score increases on the NEXT event, not the current one.
    # OR, specifically, that if home scores, the recorded 'home_score' on that row is X, 
    # and the next row is X+1 (if same game).
    
    # Actually, simplistic check: 
    # Find a game where score went 0-0 -> 1-0.
    # The Goal event row should have 0-0 state.
    
    failure_count = 0
    
    # easier: iterate meaningful sequence
    # Limit to one game for clarity
    game_id = goals.iloc[0]['game_id']
    game_df = df[df['game_id'] == game_id].sort_values('total_time_elapsed_s')
    
    print(f"Inspecting Game {game_id}...")
    
    prev_home = 0
    prev_away = 0
    
    for idx, row in game_df.iterrows():
        # Check if stored score matches our tracking
        curr_home = row['home_score']
        curr_away = row['away_score']
        
        if curr_home != prev_home or curr_away != prev_away:
            print(f"Mismatch at {idx} time {row['total_time_elapsed_s']}: Row says {curr_home}-{curr_away}, Tracking says {prev_home}-{prev_away}")
            # If this is a goal row, it SHOULD match prev.
            # If this is AFTER a goal, it should have updated.
            failure_count += 1
            
        if row['event'] == 'goal':
            team_id = row['team_id']
            # update tracking
            if team_id == row['home_id']:
                prev_home += 1
                if row['score_diff'] != (curr_home - curr_away):
                     print(f"  Goal Row: Score Diff {row['score_diff']} matches context {curr_home}-{curr_away}")
            elif team_id == row['away_id']:
                prev_away += 1
        
        if failure_count > 5:
            print("Too many mismatches.")
            break
            
    if failure_count == 0:
        print("SUCCESS: Score tracking logic validated for sample game.")
    else:
        print("FAILED: Score tracking logic mismatch.")

if __name__ == "__main__":
    verify_score()
