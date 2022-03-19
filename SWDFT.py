import numpy as np
from gtda.time_series import SingleTakensEmbedding


class SWDFT(SingleTakensEmbedding):
    """
    Sliding window discrete Fourier transform
    """
    def transform(self, X, y=None):
        embedded = super().transform(X, y)
        embedded = np.absolute(np.fft.fft(embedded))
        return embedded

