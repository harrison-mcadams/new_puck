print("DEBUG SCRIPT STARTING")
import xgboost as xgb
import numpy as np
import traceback

def run_test(name, params, use_base_margin=True):
    print(f"\n--- Test: {name} ---")
    try:
        n = 1000
        X = np.random.rand(n, 5)
        base_margin = np.random.randn(n).astype(np.float32)
        y = np.random.randint(0, 2, n)
        
        dtrain = xgb.DMatrix(X, label=y)
        if use_base_margin:
            dtrain.set_base_margin(base_margin)
            
        bst = xgb.train(params, dtrain, num_boost_round=10)
        print("SUCCESS")
        
        # Check if bias is reasonable (if gblinear)
        if params.get('booster') == 'gblinear':
            dump = bst.get_dump(dump_format='json')[0]
            # print(dump[:100]) 
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()


def main():
    print(f"XGBoost Version: {xgb.__version__}")
    
    # RIGOROUS TEST
    print("\n--- Test: Offset Logic (Target=High, Base=High) ---")
    # Case: y is all 1 (High prob). Setup base_margin to be high (logit(0.99) ~ 4.6).
    # If working: Model should learn Bias ~ 0.
    # If ignoring base: Model should learn Bias ~ 4.6.
    
    import scipy.sparse as sp
    n = 1000
    X_sp = sp.csr_matrix(np.zeros((n, 5))) # Zero features! Only Bias matters.
    y = np.ones(n) # All 1s
    
    # Base margin = 5.0 (Very high prob)
    base_margin = np.full((n, 1), 5.0, dtype=np.float32)
    
    dtrain = xgb.DMatrix(X_sp, label=y)
    dtrain.set_base_margin(base_margin)
    
    params = {
        'booster': 'gblinear',
        'objective': 'binary:logistic',
        'updater': 'coord_descent', # Test the one we use
        'learning_rate': 1.0,
        'reg_lambda': 0,
    }
    
    try:
        bst = xgb.train(params, dtrain, num_boost_round=100)
        
        # Check bias
        dump = bst.get_dump(dump_format='json')[0]
        import json
        d = json.loads(dump)
        bias = float(d.get('bias', 0))
        print(f"Learned Bias: {bias}")
        
        if abs(bias) < 1.0:
            print("RESULT: SUCCESS (Model learned residual 0)")
        elif abs(bias - 5.0) < 1.0: # If it learned ~0, it ignores base? No wait.
            # If target is 1 (inf). 
            # If base is 5. 5 is close to inf. Residual is small positive.
            # If it ignores base (assumes 0): It sees target 1, learns ~10 (large pos).
            pass
            
        # Better test: y = 0.5 (logit 0). Base = 5.0. 
        # Model should learn -5.0 to correct it back to 0.
    except:
        traceback.print_exc()

    print("\n--- Test: Offset Logic (Target=0.5, Base=5.0) ---")
    # Target 0.5 (logit 0). Base 5.0.
    # Expected: Bias ~ -5.0.
    # If ignored: Bias ~ 0.0.
    y = np.concatenate([np.zeros(500), np.ones(500)]) # Mean 0.5
    X_sp = sp.csr_matrix(np.zeros((n, 5)))
    base_margin = np.full((n, 1), 5.0, dtype=np.float32)
    
    dtrain = xgb.DMatrix(X_sp, label=y)
    dtrain.set_base_margin(base_margin)
    
    try:
        bst = xgb.train(params, dtrain, num_boost_round=100)
        dump = bst.get_dump(dump_format='json')[0]
        import json
        d = json.loads(dump)
        bias = float(d.get('bias', 0))
        print(f"Learned Bias: {bias}")
        print(f"Expected Bias: ~ -5.0")
        
        if abs(bias + 5.0) < 1.0:
             print("RESULT: SUCCESS (Offset working)")
        elif abs(bias) < 1.0:
             print("RESULT: FAIL (Offset ignored)")
        else:
             print("RESULT: ?")
    except:
        traceback.print_exc()


