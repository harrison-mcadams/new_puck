import analyze
import os

def verify_single():
    print("Verifying single player analysis...")
    out_dir = 'static/repro_phi_single'
    os.makedirs(out_dir, exist_ok=True)
    
    # Analyze Sean Couturier (8476372)
    condition = {'game_state': ['5v5'], 'is_net_empty': [0]}
    
    analyze.players(
        season='20252026',
        team='PHI',
        player_ids=[8476372],
        condition=condition,
        out_dir=out_dir
    )
    
    print(f"Single player analysis complete. Check {out_dir}")

if __name__ == "__main__":
    verify_single()
