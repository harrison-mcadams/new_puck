import analyze
import os
import shutil

def verify_filter_logic():
    print("Verifying filter logic...")
    out_dir = 'static/repro_phi_filter_logic'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Sean Couturier (8476372) - definitely played > 5 games
    # We set min_games=100 to ensure he is filtered OUT
    condition = {'game_state': ['5v5'], 'is_net_empty': [0]}
    
    print("Running with min_games=100 (expecting NO scatter plot)...")
    analyze.players(
        season='20252026',
        player_ids=[8476372],
        condition=condition,
        out_dir=out_dir,
        min_games=100
    )
    
    if os.path.exists(os.path.join(out_dir, 'player_scatter.png')):
        print("FAILURE: Scatter plot was generated despite min_games=100")
    else:
        print("SUCCESS: Scatter plot was NOT generated with min_games=100")

    # Now run with min_games=1 (expecting scatter plot)
    print("\nRunning with min_games=1 (expecting scatter plot)...")
    analyze.players(
        season='20252026',
        player_ids=[8476372],
        condition=condition,
        out_dir=out_dir,
        min_games=1
    )
    
    if os.path.exists(os.path.join(out_dir, 'player_scatter.png')):
        print("SUCCESS: Scatter plot WAS generated with min_games=1")
    else:
        print("FAILURE: Scatter plot was NOT generated with min_games=1")

if __name__ == "__main__":
    verify_filter_logic()
