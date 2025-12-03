import analyze
import os

def verify_filter():
    print("Verifying player analysis filtering (min_games=5)...")
    out_dir = 'static/repro_phi_filter'
    os.makedirs(out_dir, exist_ok=True)
    
    # Run for PHI with min_games=5
    # This will generate maps for all players but only plot scatter for those with >= 5 games
    condition = {'game_state': ['5v5'], 'is_net_empty': [0]}
    
    analyze.players(
        season='20252026',
        team='PHI',
        condition=condition,
        out_dir=out_dir,
        min_games=5
    )
    
    print(f"Analysis complete. Check {out_dir}")

if __name__ == "__main__":
    verify_filter()
