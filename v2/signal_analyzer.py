import matplotlib.pyplot as plt
import numpy as np
from PyEMD import EMD
from sqlalchemy import collate

from signals import Signal


class SignalAnalyzer:

    def __init__(self, signal: Signal) -> None:
        self.signal = signal

    def perform_emd(self, plot_emd=None):
        emd = EMD()  # initialize EMD object
        imfs = emd.emd(self.signal.y)
        if plot_emd and plot_emd is not None:
            n_imfs = len(imfs)
            y_range = (np.max(self.signal.y) - np.min(self.signal.y)) * 1.2
            plt.figure(figsize=(12, 12))
            for i_imf in range(n_imfs - 1):
                plt.subplot(n_imfs, 1, i_imf + 1)
                # plt.plot(self.signal.x, self.signal.y, color="0.8")
                plt.plot(self.signal.x, imfs[i_imf], "k")
                # plt.xticks([])
                plt.xlim([np.min(self.signal.x), np.max(self.signal.x)])
                imf_mean = np.mean(imfs[i_imf])
                plt.ylim([imf_mean - y_range / 2, imf_mean + y_range / 2])
                plt.ylabel(f"IMF {i_imf + 1}")
                plt.grid()
            plt.subplot(n_imfs, 1, n_imfs)
            plt.plot(self.signal.x, self.signal.y, color="0.8")
            plt.plot(self.signal.x, imfs[-1], "k")
            plt.xlim([min(self.signal.x), max(self.signal.x)])
            plt.ylabel("Residual")
            if self.signal.x_lbl is not None:
                plt.xlabel(self.signal.x_lbl)
            plt.grid()
            plt.tight_layout()
            plt.show()

        return imfs


# Generate a sample signal
# t = np.linspace(0, 1, 500)
# s = (
#     np.sin(2 * np.pi * 5 * t)
#     + np.sin(2 * np.pi * 15 * t)
#     + 0.5 * np.random.randn(len(t))
# )
#
# # Initialize EMD
# emd = EMD()
#
# # Decompose the signal into IMFs
# IMFs = emd.emd(s)
#
# # IMFs will be a 2D array where each row is an IMF
# # The last row is typically the residual
# print(f"Number of IMFs extracted: {IMFs.shape[0]}")
#
