
import timing
import sys

def reproduce():
    game_id = 2025020403 # Example game ID
    print(f"Testing compute_intervals_for_game with game_id={game_id} and empty condition...")
    
    try:
        # Empty condition
        res = timing.compute_intervals_for_game(game_id, condition={})
        
        print("\nResult:")
        print(f"Intersection Seconds: {res.get('intersection_seconds')}")
        print(f"Total Observed Seconds: {res.get('total_observed_seconds')}")
        print(f"Intervals Per Condition: {res.get('intervals_per_condition')}")
        
        if res.get('intersection_seconds') == 0.0 and res.get('total_observed_seconds') > 0:
            print("\nISSUE REPRODUCED: intersection_seconds is 0.0 despite total_observed_seconds > 0")
        else:
            print("\nIssue NOT reproduced.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reproduce()
