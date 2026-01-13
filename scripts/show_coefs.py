import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib

path = 'analysis/xgs/xg_model_nested.joblib'
clf = joblib.load(path)
model_fin = clf.model_finish.named_steps['clf']
preproc = clf.model_finish.named_steps['preprocessor']

print("Intercept:", model_fin.intercept_[0])
coefs = model_fin.coef_[0]

# Try getting feature names
try:
    names = preproc.get_feature_names_out()
except:
    names = [f"F{i}" for i in range(len(coefs))]

# Sort and print
pairs = sorted(zip(names, coefs), key=lambda x: x[1])

print("\n--- NEGATIVE (Bad for Goals) ---")
for n, c in pairs[:10]:
    print(f"{n}: {c:.4f}")

print("\n--- POSITIVE (Good for Goals) ---")
for n, c in pairs[-10:]:
    print(f"{n}: {c:.4f}")
    
print("\n--- DISTANCE CHECKS ---")
for n, c in pairs:
    if 'distance' in n.lower():
        print(f"{n}: {c:.4f}")
