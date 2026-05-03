
import joblib
import os
import sys

path = sys.argv[1]
if os.path.exists(path):
    model = joblib.load(path)
    print(f"Path: {path}")
    print(f"Model Type: {type(model)}")
else:
    print(f"Model not found: {path}")
