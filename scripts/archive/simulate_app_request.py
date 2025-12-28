
import analyze
import logging
import json

# Setup logging to capture output
logging.basicConfig(level=logging.DEBUG)

def simulate_app_request():
    # Mimic app.py replot logic for default inputs
    print("--- Simulating App Request ---")
    
    # Inputs as they would come from the form
    game_val = 2025020412
    game_states = [''] # "Any" selected
    net_empty = ""
    player_id = ""
    raw_condition = ""
    
    # Logic from app.py
    condition = {}
    
    # 1. Game State
    if game_states:
        if '' in game_states:
            pass
        else:
            valid_states = [s for s in game_states if s.strip()]
            if valid_states:
                condition['game_state'] = valid_states

    # 2. Net Empty
    if net_empty is not None and net_empty.strip() != '':
        try:
            condition['is_net_empty'] = [int(net_empty)]
        except Exception:
            pass

    # 3. Player ID
    if player_id and player_id.strip():
        try:
            condition['player_id'] = int(player_id)
        except Exception:
            pass

    # 4. JSON Fallback
    if not condition:
        raw_condition = (raw_condition or '{}').strip()
        try:
            if raw_condition:
                loaded = json.loads(raw_condition)
                if isinstance(loaded, dict):
                    condition = loaded
        except Exception as e:
            print(f"Failed to parse condition JSON: {e}")

    print(f"App constructed condition: {condition}")
    
    # Call analyze.xgs_map
    print("Calling analyze.xgs_map...")
    try:
        ret = analyze.xgs_map(
            game_id=game_val,
            condition=condition,
            out_path='debug_simulation_map.png',
            show=False,
            return_heatmaps=False,
            events_to_plot=['shot-on-goal', 'goal', 'xgs'],
            return_filtered_df=True,
            force_refresh=True
        )
        # xgs_map returns (out_path, heatmaps, df, summary_stats)
        if isinstance(ret, tuple) and len(ret) >= 4:
            summary_stats = ret[3]
            print("\n--- Summary Stats ---")
            for k, v in summary_stats.items():
                print(f"{k}: {v}")
        else:
            print(f"Unexpected return from xgs_map: {type(ret)}")
            
    except Exception as e:
        print(f"analyze.xgs_map failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simulate_app_request()
