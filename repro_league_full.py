
import analyze
import sys

def run_repro():
    print("--- Running League Baseline Repro ---")
    # Use a small season or mock data if possible, but here we use the real one
    # We expect this to print the DEBUG logs we added
    try:
        # We need to force 'compute' mode
        # And we pass a condition
        cond = {'game_state': ['5v5']}
        print(f"Calling analyze.league with condition={cond}")
        res = analyze.league(season='20252026', mode='compute', condition=cond)
        print("analyze.league finished")
    except Exception as e:
        print(f"analyze.league failed: {e}")

if __name__ == '__main__':
    run_repro()
