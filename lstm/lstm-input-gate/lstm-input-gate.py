import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> dict:
    """
    Returns input_gate and candidate_state as float64 arrays.
    """
    joined = np.concatenate([h_prev, x_t], axis=-1)

    i_t = sigmoid(joined @ W_i.T + b_i)
    c_tilde = np.tanh(joined @ W_c.T + b_c)

    return {
        "input_gate": i_t.astype(np.float64),
        "candidate_state": c_tilde.astype(np.float64),
    }