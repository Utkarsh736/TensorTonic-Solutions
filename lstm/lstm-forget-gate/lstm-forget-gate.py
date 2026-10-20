import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    """
    Returns the float64 forget-gate values.
    """
    joined = np.concatenate([h_prev, x_t], axis=-1)
    logits = joined @ W_f.T + b_f
    f_t = sigmoid(logits)

    return f_t.astype(np.float64)