import pandas as pd
import numpy as np
import scipy.sparse as sp
from puck import analyze, data_pipeline, mixed_effects
from puck.logistic_solver import loss_func, grad_func
from scripts.evaluate_predictive_power import split_data, get_team_game_counts, process_schedule_from_events

def main():
    df = pd.read_csv("data/20252026.csv")
    df = data_pipeline.preprocess_features(df, is_training=False, apply_imputation=True, apply_arena_adjustments=False, apply_bio_enrichment=False, apply_filtering=True)
    df, _, _ = analyze._predict_xgs(df)
    df = df[df['event'].str.lower().isin(['goal', 'shot-on-goal', 'missed-shot'])].copy()

    sched_df = process_schedule_from_events(df)
    team_schedules = get_team_game_counts(sched_df)
    train_df, _ = split_data(df, team_schedules, 15)
    
    mixed = mixed_effects.GameMixedEffectsXG(
        base_model_path="analysis/xgs/xg_model_nested_tensor.joblib",
        feature_set=[], use_tensor_splines=True, component_model_type='intercept_and_features', l2_reg=1.0 
    )
    mixed.fit(train_df)
    
    model = mixed.state_models_['5v5']
    coefs = model.get_coefficients()
    
    off_coefs = coefs[coefs['role'] == 'Offense']
    def_coefs = coefs[coefs['role'] == 'Defense']
    
    print(f"Mean Offense: {off_coefs['coef'].mean()}, Std: {off_coefs['coef'].std()}")
    print(f"Mean Defense: {def_coefs['coef'].mean()}, Std: {def_coefs['coef'].std()}")
    
    print("\n--- Top 5 Offensive Intercepts ---")
    print(off_coefs.sort_values('coef', ascending=False).head(5))
    
    print("\n--- Bottom 5 Offensive Intercepts ---")
    print(off_coefs.sort_values('coef', ascending=True).head(5))

    print("\n--- Top 5 Defensive Intercepts (Positive = Bad Defense) ---")
    print(def_coefs.sort_values('coef', ascending=False).head(5))
    
if __name__ == "__main__":
    main()
