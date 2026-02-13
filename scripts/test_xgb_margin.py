import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss

def test_base_margin_support():
    print("Testing XGBoost group-linear base_margin support...")
    
    # Simple data: y is always 1
    X = np.random.rand(100, 1)
    y = np.ones(100)
    
    # Base margin = 10 (very high logit, prob ~1)
    # If base_margin is respected, the booster should learn weights ~0.
    # If base_margin is ignored, it starts from 0 (prob 0.5) and learns positive weights.
    margin_high = np.full(100, 10.0)
    
    clf = xgb.XGBClassifier(booster='gblinear', n_estimators=100, learning_rate=0.1)
    clf.fit(X, y, base_margin=margin_high)
    
    pred_margin = clf.get_booster().predict(xgb.DMatrix(X), output_margin=True)
    print(f"Pred Margin (Delta) with High Base: {np.mean(pred_margin):.4f} (Expect ~0)")
    
    # Base margin = -10 (very low logit, prob ~0)
    # If respected, booster should learn positive weights to get to 1.
    margin_low = np.full(100, -10.0)
    clf_low = xgb.XGBClassifier(booster='gblinear', n_estimators=100, learning_rate=0.1)
    clf_low.fit(X, y, base_margin=margin_low)
    
    pred_margin_low = clf_low.get_booster().predict(xgb.DMatrix(X), output_margin=True)
    print(f"Pred Margin (Delta) with Low Base:  {np.mean(pred_margin_low):.4f} (Expect Positive)")

if __name__ == "__main__":
    test_base_margin_support()
