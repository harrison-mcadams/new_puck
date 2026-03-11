import sys
sys.path.append('.')
import joblib

me = joblib.load("analysis/xgs/joint_mixed_effects.joblib")

def print_team(team):
    print(f"\n--- {team} ---")
    for state in ['5v5', '5v4', '4v5']:
        if state in me.state_models_:
            sub = me.state_models_[state]
            if hasattr(sub, 'team_intercepts_') and team in sub.team_intercepts_:
                print(f"{state}:")
                print(f"  Offense: {sub.team_intercepts_[team]['off_intercept']:.4f}")
                print(f"  Defense: {sub.team_intercepts_[team]['def_intercept']:.4f}")
            else:
                print(f"{state}: Not found in submodel")
        else:
            print(f"{state}: No submodel")

print_team("PHI")
print_team("COL")
