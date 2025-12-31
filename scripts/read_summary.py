filename = r"c:\Users\harri\Desktop\new_puck\data\gravity_summary.txt"
try:
    with open(filename, 'r') as f:
        print(f.read())
except Exception as e:
    print(e)
