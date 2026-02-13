import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss_func(w, X, y, base_margin, l2_reg):
    # w: (n_features,)
    # X: (n_samples, n_features) sparse
    # y: (n_samples,)
    # base_margin: (n_samples,)
    
    # logits = base + Xw
    logits = base_margin + X.dot(w)
    
    # numerical stability for LogLoss
    # loss = - (y * log(p) + (1-y) * log(1-p))
    # p = sigmoid(logits)
    # Rewrite using log-sum-exp trick:
    # loss = max(logits, 0) - logits * y + log(1 + exp(-abs(logits)))
    
    # Vectorized log loss
    # np.logaddexp(0, -logits) if y=1? No.
    # standard stable binary cross entropy:
    # cost = -y * logits + log(1 + exp(logits))
    # more stable: max(logits, 0) - y * logits + log(1 + exp(-abs(logits)))
    
    term1 = np.maximum(logits, 0) - logits * y
    term2 = np.log(1 + np.exp(-np.abs(logits)))
    loss = np.sum(term1 + term2)
    
    # Regularization
    reg = 0.5 * l2_reg * np.sum(w**2)
    
    return loss + reg

def grad_func(w, X, y, base_margin, l2_reg):
    logits = base_margin + X.dot(w)
    p = sigmoid(logits)
    
    # Gradient of NLL w.r.t w is X.T @ (p - y)
    err = p - y
    grad = X.T.dot(err)
    
    # Regularization gradient
    grad += l2_reg * w
    
    return grad

def main():
    print("Testing Scipy LR with Offset...")
    
    # Simulate Data
    n_samples = 100000
    n_features = 7000
    density = 0.001
    
    print(f"Generating data: {n_samples} samples, {n_features} features, density={density}")
    t0 = time.time()
    X = sp.random(n_samples, n_features, density=density, format='csr')
    w_true = np.random.randn(n_features) * 0.5
    
    # Base margin (random offsets)
    base_margin = np.random.randn(n_samples)
    
    # Generate y based on True Model
    logits = base_margin + X.dot(w_true)
    probs = sigmoid(logits)
    y = (np.random.rand(n_samples) < probs).astype(float)
    
    print(f"Data gen took {time.time()-t0:.2f}s")
    
    # Fit
    print("Fitting...")
    w_init = np.zeros(n_features)
    l2_reg = 1.0
    
    t0 = time.time()
    # Use L-BFGS-B
    res = minimize(
        fun=loss_func,
        x0=w_init,
        args=(X, y, base_margin, l2_reg),
        jac=grad_func,
        method='L-BFGS-B',
        options={'disp': True, 'maxiter': 100}
    )
    print(f"Fitting took {time.time()-t0:.2f}s")
    print(f"Success: {res.success}, Message: {res.message}")
    print(f"Iterations: {res.nit}")
    
    # Check Bias
    # In this simulation, w_true includes everything.
    # If base_margin is used correctly, w_est should be close to w_true.
    
    corr = np.corrcoef(w_true, res.x)[0, 1]
    print(f"Correlation between True and Est weights: {corr:.4f}")
    
    # Preds
    est_logits = base_margin + X.dot(res.x)
    est_probs = sigmoid(est_logits)
    print(f"Mean Pred: {est_probs.mean():.4f}, Mean True: {y.mean():.4f}")

    # Verify Offset Logic explicitly
    # Set y to target 0.5 (logit 0). Base to 5.0. 
    # Model should minimize (0 - (5 + Offset)) -> Offset should be -5. Intercept in w.
    # But here X is random.
    # Let's add an intercept column to X? 
    # Or just rely on w to handle it if there's a constant feature.
    # In reality mixed effects model has team features (OHE), so sum(w) for a row ~ intercept.
    
if __name__ == "__main__":
    main()
