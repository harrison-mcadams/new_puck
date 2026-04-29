import sys
from . import fit_xgboost_tensor
sys.modules['puck.fit_xgboost_alternate'] = fit_xgboost_tensor
