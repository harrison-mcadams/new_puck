import sys
import os
sys.path.append(os.path.abspath('.'))
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def inspect_matchup_adjustments(home, away, state):
    model_path = Path(f"analysis/mixed_effects_heatmaps_20252026/models/mixed_model_{state}.pkl")
    bank_path = Path("analysis/mixed_effects_heatmaps_20252026/events_bank.pkl")
    
    if not model_path.exists():
        print(f"Model {model_path} not found.")
        return
    
    model = joblib.load(model_path)
    bank = joblib.load(bank_path)
    
    print(f"--- Inspecting {state} Model: {home} (Def) vs {away} (Off) ---")
    
    # Filter bank for this state
    mask = bank['game_state'] == state
    df_bank = bank[mask].copy()
    
    if len(df_bank) == 0:
        print(f"No events found in bank for state {state}")
        return

    # Use first event as individual example, but override teams
    example = df_bank.iloc[[0]].copy()
    example['team_name'] = away
    example['opp_team_name'] = home
    
    # Get base margin
    base_probs = model.base_model_.predict_proba(example)[:, 1]
    base_margin = np.log(base_probs / (1 - base_probs))[0]
    
    # Get deltas
    off_delta = model._get_layer_deltas(model.offense_layer_, example)[0]
    def_delta = model._get_layer_deltas(model.defense_layer_, example)[0]
    
    final_margin = base_margin + off_delta + def_delta
    final_prob = 1.0 / (1.0 + np.exp(-final_margin))
    
    print(f"Single Shot Example (X={example.iloc[0]['x_adj']:.0f}, Y={example.iloc[0]['y_adj']:.0f}):")
    print(f"  Base Prob:   {base_probs[0]:.4f} (Margin: {base_margin:.4f})")
    print(f"  Off Delta:   {off_delta:.4f} ({away} Adjustment)")
    print(f"  Def Delta:   {def_delta:.4f} ({home} Adjustment)")
    print(f"  Final Prob:  {final_prob:.4f}")
    
    # Now compute average for ALL bank events in this matchup context
    df_matchup = df_bank.copy()
    df_matchup['team_name'] = away
    df_matchup['opp_team_name'] = home
    
    probs = model.predict_proba(df_matchup)[:, 1]
    mean_matchup_xg = probs.mean()
    
    # For comparison, compute average with generic/league base
    base_only_probs = model.base_model_.predict_proba(df_matchup)[:, 1]
    mean_base_xg = base_only_probs.mean()
    
    print(f"\nAggregate Statistics for {len(df_bank)} shots in bank:")
    print(f"  Mean Base xG/Shot:    {mean_base_xg:.4f}")
    print(f"  Mean Matchup xG/Shot: {mean_matchup_xg:.4f}")
    print(f"  Total PP xG (6.5 shots): {mean_matchup_xg * 6.5:.2f}")

if __name__ == "__main__":
    # PHI PP in simulation uses '4v5' model
    inspect_matchup_adjustments("COL", "PHI", "4v5")
