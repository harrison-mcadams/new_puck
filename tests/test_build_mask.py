import sys
from parse import build_mask
import pandas as pd

sys.stdout.write('=== Test 1: mixed types in column ===\n')
sys.stdout.flush()
df = pd.DataFrame({'is_net_empty':[0,1,'0','1',None,'', True, False], 'x':list(range(1,9))})
sys.stdout.write(df.to_string() + '\n')
sys.stdout.flush()
cond = {'is_net_empty':[0,1]}
mask = build_mask(df, cond)
sys.stdout.write('\nMask values:\n')
sys.stdout.write(str(mask.to_list()) + '\n')
sys.stdout.write('\nMatched rows:\n')
sys.stdout.write(df[mask].to_string() + '\n')
sys.stdout.flush()

sys.stdout.write('\n=== Test 2: column as strings ===\n')
sys.stdout.flush()
df2 = pd.DataFrame({'is_net_empty':['0','1','2','',None], 'x':[1,2,3,4,5]})
sys.stdout.write(df2.to_string() + '\n')
mask2 = build_mask(df2, cond)
sys.stdout.write('\nMask2 values:\n')
sys.stdout.write(str(mask2.to_list()) + '\n')
sys.stdout.write('\nMatched rows in DF2:\n')
sys.stdout.write(df2[mask2].to_string() + '\n')
sys.stdout.flush()

sys.stdout.write('\n=== Test 3: camelCase key normalization simulation ===\n')
sys.stdout.flush()
mask3 = build_mask(df, {'is_net_empty': [0,1]})
sys.stdout.write('Mask3 values (using corrected key):\n')
sys.stdout.write(str(mask3.to_list()) + '\n')
sys.stdout.write('\nMatched rows for mask3:\n')
sys.stdout.write(df[mask3].to_string() + '\n')
sys.stdout.flush()
