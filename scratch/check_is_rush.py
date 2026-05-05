import joblib
import os

model_path = 'analysis/xgs/xg_model_xgboost_tensor_modern_era.joblib'
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    exit(1)

m = joblib.load(model_path)
for sub in ['block', 'acc', 'fin']:
    booster = getattr(m, 'model_'+sub).get_booster()
    importance = booster.get_score(importance_type='weight').get('is_rush', 0)
    print(f"{sub} is_rush weight: {importance}")
    
    # Check split conditions
    dump = booster.get_dump()
    splits = []
    for tree in dump:
        if 'is_rush' in tree:
            # Extract split condition
            import re
            matches = re.findall(r'is_rush<([\d\.]+)', tree)
            splits.extend(matches)
    if splits:
        print(f"  {sub} is_rush splits: {set(splits)}")
