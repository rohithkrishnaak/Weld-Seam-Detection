"""
Fuzzy Morphological Operations for Laser Stripe Cleaning.

Uses fuzzy structuring elements with gradual edges to preserve
stripe structure while removing noise.

References:
    [1] Bloch, I. & Maitre, H. (1995). Fuzzy mathematical morphologies.
        Pattern Recognition, 28(9), 1341-1387.
    [2] De Baets, B. (1997). Fuzzy morphology: A contribution to the
        analysis of fuzzy images. PhD Thesis.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class FuzzyMorphology:
    """Fuzzy morphological operations using a 1-D fuzzy structuring element.

    Unlike classical morphology where the structuring element (SE) is
    binary, a *fuzzy* SE has values in [0, 1] that represent the degree
    to which each position participates in the operation.  This allows
    gradual transitions at the stripe edges instead of hard cut-offs.

    Parameters
    ----------
    kernel : array-like, shape (K,)
        1-D fuzzy structuring element with values in [0, 1].
        Example: ``[0.2, 0.5, 1.0, 1.0, 1.0, 0.5, 0.2]``.
    """

    def __init__(self, kernel: np.ndarray | list[float]) -> None:
        self.kernel: np.ndarray = np.asarray(kernel, dtype=np.float64)
        if self.kernel.ndim != 1:
            raise ValueError("Kernel must be a 1-D array.")
        # Validate range
        if self.kernel.min() < 0.0 or self.kernel.max() > 1.0:
            raise ValueError("Kernel values must lie in [0, 1].")

    # ------------------------------------------------------------------
    # Primitive operations (1-D)
    # ------------------------------------------------------------------
    def fuzzy_dilate(self, signal: np.ndarray) -> np.ndarray:
        """Fuzzy dilation of a 1-D signal.

        Definition (Bloch & Maitre 1995, eq. 7):
            (A ⊕_F B)(x) = max_y  min( A(y),  B(x − y) )

        We slide the (flipped) kernel over the signal and at each
        position take max of element-wise min.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Input signal with values in [0, 1].

        Returns
        -------
        dilated : np.ndarray, shape (N,)
        """
        signal = np.asarray(signal, dtype=np.float64)
        k_len = len(self.kernel)
        half_k = k_len // 2

        # Reflect-pad to preserve signal length
        padded = np.pad(signal, pad_width=half_k, mode="reflect")

        # Create a sliding window view: shape (N, K)
        windows = sliding_window_view(padded, window_shape=k_len)

        # Fuzzy dilation: max_y min(A(y), B(x-y))
        # The kernel is applied in the natural (non-flipped) order
        # because dilation uses B(x-y) which mirrors the kernel.
        # sliding_window_view already provides the aligned windows.
        dilated = np.max(np.minimum(windows, self.kernel[np.newaxis, :]), axis=1)

        return dilated

    def fuzzy_erode(self, signal: np.ndarray) -> np.ndarray:
        """Fuzzy erosion of a 1-D signal.

        Definition (Bloch & Maitre 1995, eq. 8):
            (A ⊖_F B)(x) = min_y  max( A(y),  1 − B(y − x) )

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Input signal with values in [0, 1].

        Returns
        -------
        eroded : np.ndarray, shape (N,)
        """
        signal = np.asarray(signal, dtype=np.float64)
        k_len = len(self.kernel)
        half_k = k_len // 2

        # Reflect-pad
        padded = np.pad(signal, pad_width=half_k, mode="reflect")

        # Sliding windows: (N, K)
        windows = sliding_window_view(padded, window_shape=k_len)

        # Complement of the kernel: 1 − B
        kernel_complement = 1.0 - self.kernel  # shape (K,)

        # Fuzzy erosion: min_y max(A(y), 1 − B(y − x))
        eroded = np.min(
            np.maximum(windows, kernel_complement[np.newaxis, :]), axis=1
        )

        return eroded

    # ------------------------------------------------------------------
    # Composite operations (1-D)
    # ------------------------------------------------------------------
    def fuzzy_open(self, signal: np.ndarray) -> np.ndarray:
        """Fuzzy opening: erosion followed by dilation.

        Opening removes bright noise speckles narrower than the
        structuring element while preserving broader structures.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)

        Returns
        -------
        opened : np.ndarray, shape (N,)
        """
        return self.fuzzy_dilate(self.fuzzy_erode(signal))

    def fuzzy_close(self, signal: np.ndarray) -> np.ndarray:
        """Fuzzy closing: dilation followed by erosion.

        Closing fills narrow dark gaps while preserving broader
        structures.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)

        Returns
        -------
        closed : np.ndarray, shape (N,)
        """
        return self.fuzzy_erode(self.fuzzy_dilate(signal))

    # ------------------------------------------------------------------
    # 2-D stripe cleaning (row-wise application)
    # ------------------------------------------------------------------
    def clean_stripe(self, membership_map: np.ndarray) -> np.ndarray:
        """Clean a 2-D membership map by applying fuzzy opening + closing.

        The 1-D kernel is applied along each *row* (horizontal
        direction), which is the typical orientation of noise variation
        in vertically oriented laser stripe images.

        Workflow:
            1. Fuzzy opening (row-wise) — removes narrow bright speckles
            2. Fuzzy closing (row-wise) — fills narrow dark gaps

        Parameters
        ----------
        membership_map : np.ndarray, shape (H, W)
            Laser-cluster membership image with values in [0, 1].

        Returns
        -------
        cleaned : np.ndarray, shape (H, W)
            Cleaned membership map.
        """
        membership_map = np.asarray(membership_map, dtype=np.float64)

        if membership_map.ndim == 1:
            # Scalar 1-D case — apply directly
            return self.fuzzy_close(self.fuzzy_open(membership_map))

        h, w = membership_map.shape
        cleaned = np.empty_like(membership_map)

        # --- Vectorised row-wise processing ---
        # We process all rows simultaneously by reshaping the padded
        # image into (H, W + 2*pad) and using stride tricks.
        k_len = len(self.kernel)
        half_k = k_len // 2

        # Step 1: fuzzy opening (erode, then dilate) along rows
        # Pad each row via reflect
        padded = np.pad(
            membership_map,
            pad_width=((0, 0), (half_k, half_k)),
            mode="reflect",
        )  # (H, W + 2*half_k)

        # Sliding windows over columns for each row: (H, W, K)
        windows = sliding_window_view(padded, window_shape=k_len, axis=1)

        # Erosion: min_y max(A(y), 1 − B(y − x))
        kernel_c = 1.0 - self.kernel  # (K,)
        eroded = np.min(
            np.maximum(windows, kernel_c[np.newaxis, np.newaxis, :]), axis=2
        )  # (H, W)

        # Dilation of the eroded result
        padded_e = np.pad(
            eroded, pad_width=((0, 0), (half_k, half_k)), mode="reflect"
        )
        windows_e = sliding_window_view(padded_e, window_shape=k_len, axis=1)
        opened = np.max(
            np.minimum(windows_e, self.kernel[np.newaxis, np.newaxis, :]),
            axis=2,
        )  # (H, W)

        # Step 2: fuzzy closing (dilate, then erode) on opened result
        padded_o = np.pad(
            opened, pad_width=((0, 0), (half_k, half_k)), mode="reflect"
        )
        windows_o = sliding_window_view(padded_o, window_shape=k_len, axis=1)
        dilated = np.max(
            np.minimum(windows_o, self.kernel[np.newaxis, np.newaxis, :]),
            axis=2,
        )  # (H, W)

        padded_d = np.pad(
            dilated, pad_width=((0, 0), (half_k, half_k)), mode="reflect"
        )
        windows_d = sliding_window_view(padded_d, window_shape=k_len, axis=1)
        cleaned = np.min(
            np.maximum(windows_d, kernel_c[np.newaxis, np.newaxis, :]), axis=2
        )  # (H, W)

        return cleaned
