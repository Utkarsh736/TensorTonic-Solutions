import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def lstm_cell(x_t, h_prev, C_prev,
              W_f, W_i, W_c, W_o,
              b_f, b_i, b_c, b_o):
    joined = np.concatenate([h_prev, x_t], axis=-1)
    f_t = sigmoid(joined @ W_f.T + b_f)
    i_t = sigmoid(joined @ W_i.T + b_i)
    c_tilde = np.tanh(joined @ W_c.T + b_c)
    C_t = f_t * C_prev + i_t * c_tilde
    o_t = sigmoid(joined @ W_o.T + b_o)
    h_t = o_t * np.tanh(C_t)
    return h_t, C_t

def lstm_forward(X: np.ndarray, W_f: np.ndarray, W_i: np.ndarray,
                 W_c: np.ndarray, W_o: np.ndarray, b_f: np.ndarray,
                 b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray,
                 W_y: np.ndarray, b_y: np.ndarray) -> dict:
    """
    Returns outputs, final_hidden_state, and final_cell_state.
    """
    N, T, D = X.shape
    H = W_f.shape[0]
    O = W_y.shape[0]

    h = np.zeros((N, H), dtype=np.float64)
    C = np.zeros((N, H), dtype=np.float64)
    outputs = []

    for t in range(T):
        x_t = X[:, t, :]                    # (N, D)
        h, C = lstm_cell(x_t, h, C,
                         W_f, W_i, W_c, W_o,
                         b_f, b_i, b_c, b_o)
        y_t = h @ W_y.T + b_y              # (N, O)
        outputs.append(y_t)

    outputs = np.stack(outputs, axis=1)     # (N, T, O)

    return {
        "outputs": outputs.astype(np.float64),
        "final_hidden_state": h.astype(np.float64),
        "final_cell_state": C.astype(np.float64),
    }