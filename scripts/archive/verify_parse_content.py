
import os

path = r'c:\Users\harri\Desktop\new_puck\puck\parse.py'
print(f"Reading {path}")
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
start = 1445
end = 1460
print(f"Lines {start}-{end}:")
for i in range(start, end):
    if i < len(lines):
        print(f"{i+1}: {lines[i].rstrip()}")
