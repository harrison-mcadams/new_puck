
import os
import sys
import numpy as np
import pandas as pd
import logging
from scipy.stats import poisson

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.evaluate_predictive_power import PoissonMatchupEngine

def test_poisson_sensitivity():
    engine_mult = PoissonMatchupEngine(logic_type='multiplicative')
    engine_add = PoissonMatchupEngine(logic_type='additive')

    # Case: Strong Team (90th percentile) vs Weak Team (10th percentile)
    # Using stats from previous run
    # xG: Mean 3.18, Std 0.27
    # Actual: Mean 2.98, Std 0.34
    
    # xG Stats
    strong_xg = 3.18 + 1.28 * 0.27 # ~3.53
    weak_xg = 3.18 - 1.28 * 0.27   # ~2.83
    league_xg = 3.18
    
    # Actual Stats
    strong_act = 2.98 + 1.28 * 0.34 # ~3.42
    weak_act = 2.98 - 1.28 * 0.34   # ~2.54
    league_act = 2.98

    def get_prob(engine, h, a, l):
        abilities = {
            'home': {'for': h, 'ag': l}, # Assume average defense
            'away': {'for': a, 'ag': l}
        }
        return engine.predict_winner_prob('home', 'away', abilities, l)

    print("--- Sensitivity Analysis: Strong (90th) vs Weak (10th) Offense ---")
    print(f"xG Case (Strong {strong_xg:.2f} vs Weak {weak_xg:.2f}, Lg {league_xg:.2f})")
    print(f"  Multiplicative Prob: {get_prob(engine_mult, strong_xg, weak_xg, league_xg):.4f}")
    print(f"  Additive Prob:       {get_prob(engine_add, strong_xg, weak_xg, league_xg):.4f}")
    
    print(f"\nActual Case (Strong {strong_act:.2f} vs Weak {weak_act:.2f}, Lg {league_act:.2f})")
    print(f"  Multiplicative Prob: {get_prob(engine_mult, strong_act, weak_act, league_act):.4f}")
    print(f"  Additive Prob:       {get_prob(engine_add, strong_act, weak_act, league_act):.4f}")

    # What if we "scale up" xG variance to match Actual?
    scaled_std = 0.34
    strong_xg_scaled = 3.18 + 1.28 * scaled_std
    weak_xg_scaled = 3.18 - 1.28 * scaled_std
    
    print(f"\nxG Case SCALED VARIANCE (Strong {strong_xg_scaled:.2f} vs Weak {weak_xg_scaled:.2f})")
    print(f"  Multiplicative Prob: {get_prob(engine_mult, strong_xg_scaled, weak_xg_scaled, league_xg):.4f}")
    print(f"  Additive Prob:       {get_prob(engine_add, strong_xg_scaled, weak_xg_scaled, league_xg):.4f}")

if __name__ == "__main__":
    test_poisson_sensitivity()
