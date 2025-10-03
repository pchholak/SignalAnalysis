import os

from signals import Signal

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# from signal_analyzer import SignalAnalyzer
# from signals import DuplicatePolicy, PreprocessConfig, Signal


# Read data from a list
sig = Signal([1, 2, 3])

# Read data from a list with dt index
sig = Signal(
    [1, 2, 3],
    ["2025-05-05 09:00", "2025-05-06 09:00", "2025-05-07 09:00"],
    x_as_datetime=True,
)
print(f"Index dtype: {sig.series.index.dtype}")

# # Read weight from manually entered weight data in Excel
# fpath_weight = os.path.expanduser("~/data/weight.xlsx")
# sig_1 = Signal.from_dataframe(
#     fpath_weight, "Weight", "Date", parse_x_as_datetime=True, datetime_format="%m/%d/%y"
# )
#
# # Read weight from VeSync data
# fpath_weight = os.path.expanduser("~/data/VeSync Weight Data 7-1-2025 to 9-27-2025.csv")
# sig_2 = Signal.from_dataframe(
#     fpath_weight,
#     "Weight",
#     "Time",
#     parse_x_as_datetime=True,
#     datetime_format="%m/%d/%Y, %I:%M %p",
#     regex_y=r"^(.+)lb",
# )
#
# # print(sig_df_1.y)
# # print(sig_df_2.y)
# # print(np.concatenate([sig_df_1.y, sig_df_2.y]))
#
#
# # Combine both signals
# sig = Signal.concat(
#     sig_1, sig_2, merge_x=True, x_as_datetime=True, y_lbl="Weight", x_lbl="Time"
# )
# # sig.preprocess(PreprocessConfig(duplicates=DuplicatePolicy.LAST), inplace=True)
# sig.preprocess(PreprocessConfig(duplicates=DuplicatePolicy.MEAN), inplace=True)
# for x, y in zip(sig.x, sig.y):
#     print(x, y)
#
# # Perform EMD and plot
# # sig.interpolate(k=1)
# sig_anal = SignalAnalyzer(sig)
# imfs = sig_anal.perform_emd(plot_emd=True)
