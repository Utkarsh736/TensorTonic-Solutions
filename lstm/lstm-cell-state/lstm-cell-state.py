import numpy as np

def update_cell_state(C_prev: np.ndarray, f_t: np.ndarray,
                      i_t: np.ndarray, c_tilde: np.ndarray) -> np.ndarray:
    """
    Returns the float64 updated cell state.
    """
    C_t = f_t * C_prev + i_t * c_tilde
    return C_t.astype(np.float64)