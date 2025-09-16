import numpy as np
import pandas as pd


class Signal:
    def __init__(self, sig: list | np.ndarray) -> None:
        self.signal = sig

    @classmethod
    def from_dataframe(cls, fpath_df: str, col_name: str):
        df = pd.read_excel(fpath_df)
        sig = df[col_name]
        return cls(sig.to_list())


# Signal.from_dataframe(fpath_df, col_name)
