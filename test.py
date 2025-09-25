import os

from signals import Signal

sig = Signal([1, 2, 3])
print(sig.signal)

import pandas as pd

fpath_weight = os.path.expanduser("~/data/weight.xlsx")
df = pd.read_excel(fpath_weight)
print(df)
