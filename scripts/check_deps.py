import sys
import subprocess
import importlib

required_packages = [
    'pandas',
    'numpy',
    'matplotlib',
    'requests',
    'joblib',
    'sklearn',
    'flask',
    'scipy'
]

missing = []
for pkg in required_packages:
    try:
        importlib.import_module(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"Missing packages: {', '.join(missing)}")
    print("Installing missing packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
else:
    print("All required packages are installed.")
