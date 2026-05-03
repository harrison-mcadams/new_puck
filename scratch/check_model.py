
import joblib
import os

model_path = os.path.join('analysis', 'xgs', 'xg_model_nested_tensor.joblib')
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"Model Type: {type(model)}")
    print(f"Attributes: {dir(model)}")
else:
    print("Model not found.")
