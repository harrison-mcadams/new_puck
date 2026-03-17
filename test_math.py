
import numpy as np

def test_math():
    league_avg = 3.1
    h_for = 3.5
    h_ag = 2.7
    a_for = 3.0
    a_ag = 3.2
    
    # New Multiplicative Logic: (H_For * A_Ag) / League_Avg
    h_exp_m = (h_for * a_ag) / league_avg
    a_exp_m = (a_for * h_ag) / league_avg
    
    # Old Additive Logic: League_Avg + (H_GD - A_GD) / 2
    # H_GD = H_For - H_Ag
    # A_GD = A_For - A_Ag
    h_gd = h_for - h_ag
    a_gd = a_for - a_ag
    h_exp_a = league_avg + (h_gd - a_gd) / 2
    a_exp_a = league_avg + (a_gd - h_gd) / 2
    
    print(f"Rates: H_For={h_for}, H_Ag={h_ag}, A_For={a_for}, A_Ag={a_ag}, Avg={league_avg}")
    print(f"Multiplicative: H_Exp={h_exp_m:.4f}, A_Exp={a_exp_m:.4f}")
    print(f"Additive:       H_Exp={h_exp_a:.4f}, A_Exp={a_exp_a:.4f}")

if __name__ == "__main__":
    test_math()
