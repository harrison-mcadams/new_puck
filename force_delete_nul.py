import os

# Use the Windows-specific extended-length path prefix for the reserved file 'nul'
path = r"\\?\c:\Users\harri\Desktop\new_puck\nul"

try:
    if os.path.exists(path):
        print(f"File {path} exists. Attempting to remove...")
        os.remove(path)
        print("Success: nul file removed.")
    else:
        print(f"File {path} does not exist according to os.path.exists.")
        # Try to remove it anyway just in case the check is failing due to reservatons
        os.remove(path)
        print("Success: nul file removed (even though path check skipped/failed).")
except Exception as e:
    print(f"Error: {e}")
