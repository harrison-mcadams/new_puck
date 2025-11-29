import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze

def test_flyers_players():
    print("Testing analyze.players for Philadelphia Flyers...")
    
    season = '20252026'
    team = None # 'PHI'
    player_ids = [8480015] # Travis Konecny (example ID from previous run)
    condition = {'game_state': ['5v5'], 'is_net_empty': [0]}
    out_dir = 'static/test_flyers_players'
    
    # Clean output dir
    if os.path.exists(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        analyze.players(
            season=season,
            team=team,
            player_ids=player_ids,
            time_scope='season',
            condition=condition,
            out_dir=out_dir
        )
        print("analyze.players completed successfully.")
    except Exception as e:
        print(f"analyze.players failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Verify outputs
    expected_files = [
        'player_stats.csv',
        'player_scatter.png'
    ]
    
    missing = []
    for f in expected_files:
        if not os.path.exists(os.path.join(out_dir, f)):
            missing.append(f)
    
    if missing:
        print(f"FAILED: Missing output files: {missing}")
        sys.exit(1)
    
    # Check for at least one map
    maps = [f for f in os.listdir(out_dir) if f.endswith('_map.png')]
    if not maps:
        print("FAILED: No player maps generated.")
        sys.exit(1)
        
    print(f"SUCCESS: Generated {len(maps)} player maps and summary stats.")

if __name__ == '__main__':
    test_flyers_players()
