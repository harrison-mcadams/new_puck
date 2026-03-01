import pandas as pd
from puck import analyze, data_pipeline, mixed_effects
from scripts.evaluate_predictive_power import split_data, get_team_game_counts, process_schedule_from_events

def main():
    print("Loading data...")
    df = pd.read_csv("data/20252026.csv")
    df = data_pipeline.preprocess_features(df, is_training=False, apply_imputation=True, apply_arena_adjustments=False, apply_bio_enrichment=False, apply_filtering=True)
    df, _, _ = analyze._predict_xgs(df)

    sched_df = process_schedule_from_events(df)
    team_schedules = get_team_game_counts(sched_df)
    
    # Train heavily on 40 games
    train_df, _ = split_data(df, team_schedules, 15)
    
    print(f"Training on {len(train_df)} events...")
    mixed = mixed_effects.GameMixedEffectsXG(
        base_model_path="analysis/xgs/xg_model_nested_tensor.joblib",
        feature_set=[], 
        use_tensor_splines=True, 
        component_model_type='intercept_and_features',
        l2_reg=1.0 
    )
    mixed.fit(train_df)
    
    # Dump 5v5 intercepts
    model_5v5 = mixed.state_models_['5v5']
    coefs = model_5v5.get_coefficients()
    
    # Show the extreme intercepts
    off_coefs = coefs[coefs['role'] == 'Offense']
    def_coefs = coefs[coefs['role'] == 'Defense']
    
    print("\n--- Top 5 Offensive Intercepts ---")
    print(off_coefs.sort_values(by='coef', ascending=False).head(5))
    
    print("\n--- Bottom 5 Offensive Intercepts ---")
    print(off_coefs.sort_values(by='coef', ascending=True).head(5))

    print("\n--- Top 5 Defensive Intercepts (Positive = Bad Defense) ---")
    print(def_coefs.sort_values(by='coef', ascending=False).head(5))

if __name__ == "__main__":
    main()
