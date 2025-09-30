import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline


class DuplicatePolicy(Enum):
    ERROR = "error"  # raise if duplicates
    FIRST = "first"  # keep first occurrence
    LAST = "last"  # keep first occurrence
    MEAN = "mean"  # average y for duplicate x


@dataclass
class PreprocessConfig:
    sort_x: bool = True
    duplicates: DuplicatePolicy = DuplicatePolicy.ERROR


class Signal:
    def __init__(
        self,
        y: list | np.ndarray,
        x: Optional[list | np.ndarray] = None,
        x_as_datetime: bool = False,
        y_lbl: Optional[str] = None,
        x_lbl: Optional[str] = None,
    ) -> None:
        """
        Initialize the Signal object.
        """
        # Read signal `y` as an np array
        y_arr = np.array(y)
        if y_arr.ndim != 1:
            raise ValueError("y must be 1-D")

        # Read index `x` as an np array
        if x is None:
            x_arr = np.arange(y_arr.size)
        else:
            x_arr = np.array(x)
        if (x_arr.ndim != 1) or (x_arr.size != y_arr.size):
            raise ValueError("x must be 1-D and same length as y")

        # Store x and y in object
        self.x = x_arr
        self.y = y_arr

        # Keep copies of orig x and y (before preprocessing/interpolating them)
        self.y_orig = self.y.copy()
        self.x_orig = self.x.copy()

        # Assign remaining initialization values as supplied
        self.x_as_datetime = x_as_datetime
        self.y_lbl = y_lbl
        self.x_lbl = x_lbl
        self.date_interpolated = False

    def preprocess(self, cfg: PreprocessConfig, inplace: bool = False):
        x = self.x.copy()
        y = self.y.copy()

        if cfg.sort_x:
            sort_inds = np.argsort(x)
            x = x[sort_inds]
            y = y[sort_inds]

        if cfg.duplicates != DuplicatePolicy.ERROR:
            df = pd.DataFrame({"x": x, "y": y})
            if cfg.duplicates == DuplicatePolicy.FIRST:
                df = df.drop_duplicates(subset="x", keep="first", inplace=False)
            elif cfg.duplicates == DuplicatePolicy.LAST:
                df = df.drop_duplicates(subset="x", keep="last", inplace=False)
            elif cfg.duplicates == DuplicatePolicy.MEAN:
                df = df.groupby("x", as_index=False)["y"].mean()
            x = df["x"].to_numpy()
            y = df["y"].to_numpy()
        else:
            if np.unique(x).size != x.size:
                raise ValueError(
                    "Duplicate x values found; set duplicates policy to 'first', 'last', or 'mean'"
                )

        # Return in place or as an independent obj
        if inplace:
            self.x, self.y = x, y
            return self
        else:
            return Signal(
                y=y,
                x=x,
                x_as_datetime=self.x_as_datetime,
                y_lbl=self.y_lbl,
                x_lbl=self.x_lbl,
            )

    @classmethod
    def concat(
        cls,
        left,
        right,
        merge_x: bool = False,
        x_as_datetime: bool = False,
        y_lbl: Optional[str] = None,
        x_lbl: Optional[str] = None,
    ):
        y = np.concatenate([left.y, right.y])
        if merge_x:
            x = np.concatenate([left.x, right.x])
            if x_as_datetime:
                x = np.concatenate([left.x, right.x], dtype="datetime64")
        else:
            x = None
        return cls(
            y,
            x,
            x_as_datetime=x_as_datetime,
            y_lbl=y_lbl,
            x_lbl=x_lbl,
        )

    def plot_signal(self, ax: Optional[plt.Axes] = None):
        ax = ax or plt.gca()
        if self.date_interpolated:
            ax.plot(self.x_orig, self.y_orig, "o")
            ax.plot(self.x, self.y, "-.")
        else:
            ax.plot(self.x, self.y, "-o")
        if self.x_lbl is not None:
            ax.set_xlabel(self.x_lbl)
        if self.y_lbl is not None:
            ax.set_ylabel(self.y_lbl)
        return ax

    def interpolate(self, res_multiplier: float = 1, k: int = 3):
        """
        Interpolates signal `y` over `x` and returns `y_inter` interpolated over `x_interp`,
        which is sampled at the minimum resolution of `x` times `res_multiplier`.
        """
        if not self.x_as_datetime:
            x = self.x.copy()
        else:
            x = convert_npdatetime64_to_secondssinceepoch(self.x)
        y = self.y.copy()
        spline = make_interp_spline(x, y, k=k)
        min_resolution_x = min(np.diff(x))
        res_resamp = res_multiplier * min_resolution_x
        x_interp = np.arange(x[0], x[-1] + res_resamp, res_resamp)
        y_interp = spline(x_interp)
        self.y = y_interp
        if self.x_as_datetime:
            self.x = convert_secondssinceepoch_to_npdatetime64(x_interp)
        else:
            self.x = x_interp
        self.date_interpolated = True

    @classmethod
    def from_dataframe(
        cls,
        fpath_df: str,
        col_name_y: str,
        col_name_x: Optional[str] = None,
        parse_x_as_datetime: bool = False,
        datetime_format: str = "%d/%m/%Y",
        regex_y: Optional[str] = None,
    ):
        """
        Create a Signal instance from a dataframe read from Excel.

        Args:
            fpath_df: Path to the Excel file.
            col_name_y: Column name for y values.
            col_name_x: Optional column name for x values.
            parse_x_as_datetime: If True, attempt to parse x column as datetime.

        Returns:
            Signal instance.
        """
        _, file_ext = os.path.splitext(fpath_df)
        if file_ext == ".xlsx":
            df = pd.read_excel(fpath_df)
        elif file_ext == ".csv":
            df = pd.read_csv(fpath_df)
        else:
            raise ValueError(f"Unable to read dataframe: {fpath_df}")
        if regex_y is None:
            y = df[col_name_y].to_numpy()
        else:
            y = np.array(
                [float(w) for w in df[col_name_y].str.extract(regex_y).to_numpy()]
            )
        if col_name_x is None:
            x = None
        else:
            if parse_x_as_datetime:
                x = pd.to_datetime(df[col_name_x], format=datetime_format).to_numpy()
            else:
                x = df[col_name_x].to_numpy()
        return cls(
            y,
            x,
            x_as_datetime=parse_x_as_datetime,
            y_lbl=col_name_y,
            x_lbl=col_name_x,
        )


def convert_npdatetime64_to_secondssinceepoch(npdt64_series):
    return (
        np.array(
            [w.astype(np.timedelta64) / np.timedelta64(1, "ms") for w in npdt64_series]
        )
        / 1000
    )


def convert_secondssinceepoch_to_npdatetime64(timestamps):
    return np.array([datetime.fromtimestamp(ts) for ts in timestamps])
