import numpy as np

def candidate_hidden(r_t: np.ndarray, h_prev: np.ndarray, x_t: np.ndarray,
                     W_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Returns the float64 candidate hidden state.
    """
    gated_h = r_t * h_prev                                   # (H,) or (N, H)
    joined = np.concatenate([gated_h, x_t], axis=-1)          # (H+D,) or (N, H+D)
    affine = joined @ W_h.T + b_h                             # (H,) or (N, H)
    h_tilde = np.tanh(affine)

    return np.asarray(h_tilde, dtype=np.float64)