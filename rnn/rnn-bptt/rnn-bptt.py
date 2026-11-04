import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray,
                     h_prev: np.ndarray, x_t: np.ndarray,
                     W_hh: np.ndarray, W_xh: np.ndarray) -> dict:
    """
    Returns the five float64 local RNN gradients.
    """
    da = dh_next * (1.0 - h_t ** 2)

    dh_prev = da @ W_hh
    dx      = da @ W_xh

    dW_hh = da.T @ h_prev
    dW_xh = da.T @ x_t

    db_h = da.sum(axis=0)

    return {
        "dh_prev": dh_prev,
        "dx":      dx,
        "dW_hh":   dW_hh,
        "dW_xh":   dW_xh,
        "db_h":    db_h,
    }