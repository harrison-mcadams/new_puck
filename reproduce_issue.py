import analyze
import sys

def reproduce():
    # Known completed game ID from games_by_team.json (Oct 10, 2025)
    game_id = '2025020021' 
    
    print(f"Checking game status for {game_id}...")
    
    try:
        # Call xgs_map to get summary stats
        # We don't need the plot, so we can use a dummy out_path
        out_path, _, _, summary_stats = analyze.xgs_map(
            game_id=game_id,
            out_path='reproduce_test.png',
            show=False,
            return_heatmaps=True
        )
        
        game_ongoing = summary_stats.get('game_ongoing')
        time_remaining = summary_stats.get('time_remaining')
        
        print(f"Game Ongoing: {game_ongoing}")
        print(f"Time Remaining: {time_remaining}")
        
        if game_ongoing:
            print("FAILURE: Game is marked as ongoing but should be Final.")
            sys.exit(1)
        else:
            print("SUCCESS: Game is correctly marked as not ongoing.")
            sys.exit(0)
            
    except Exception as e:
        print(f"Error running reproduction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    reproduce()
