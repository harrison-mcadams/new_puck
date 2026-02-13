import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)

def sigmoid(x):
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

def loss_func(w, X, y, base_margin, l2_reg):
    """
    Computes the regularized negative log-likelihood (log loss).
    
    Args:
        w: Weight vector (n_features,)
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        base_margin: Offset vector (n_samples,) representing log-odds
        l2_reg: L2 regularization strength (lambda)
        
    Returns:
        loss: Scalar float
    """
    # Calculate logits = base + Xw
    logits = base_margin + X.dot(w)
    
    # Stable Log Loss calculation via log-sum-exp trick
    # loss = max(logits, 0) - y * logits + log(1 + exp(-abs(logits)))
    term1 = np.maximum(logits, 0) - logits * y
    term2 = np.log(1 + np.exp(-np.abs(logits)))
    loss = np.sum(term1 + term2)
    
    # L2 Regularization
    # convention: 0.5 * lambda * ||w||^2
    reg = 0.5 * l2_reg * np.sum(w**2)
    
    return loss + reg

def grad_func(w, X, y, base_margin, l2_reg):
    """
    Computes the gradient of the loss function w.r.t weights.
    
    Returns:
        grad: Gradient vector (n_features,)
    """
    logits = base_margin + X.dot(w)
    p = sigmoid(logits)
    
    # Gradient of NLL w.r.t w is X.T @ (p - y)
    err = p - y
    grad = X.T.dot(err)
    
    # Regularization gradient
    grad += l2_reg * w
    
    return grad

def fit_logistic_offset(X, y, base_margin, l2_reg=1.0, verbose=False):
    """
    Fits a logistic regression model with a fixed base_margin offset using L-BFGS-B.
    
    Args:
        X: Feature matrix (n_samples, n_features) - Sparse or Dense
        y: Target vector
        base_margin: Offset vector (log-odds)
        l2_reg: L2 regularization strength
        
    Returns:
        coef: Learned coefficients (n_features,)
    """
    n_features = X.shape[1]
    w_init = np.zeros(n_features)
    
    if verbose:
        logger.info(f"Fitting Logistic Offset Model. Samples: {X.shape[0]}, Features: {n_features}")
    
    # Use L-BFGS-B optimizer
    res = minimize(
        fun=loss_func,
        x0=w_init,
        args=(X, y, base_margin, l2_reg),
        jac=grad_func,
        method='L-BFGS-B',
        options={'disp': verbose, 'maxiter': 200}
    )
    
    if not res.success:
        logger.warning(f"Optimizer did not converge: {res.message}")
        
    return res.x
