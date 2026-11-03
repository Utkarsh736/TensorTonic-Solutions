import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray, W_xh: np.ndarray,
                W_hh: np.ndarray, b_h: np.ndarray) -> dict:
    """
    Returns hidden_states and final_hidden_state as float64 arrays.
    """
    X = np.asarray(X, dtype=np.float64)
    h_0 = np.asarray(h_0, dtype=np.float64)
    W_xh = np.asarray(W_xh, dtype=np.float64)
    W_hh = np.asarray(W_hh, dtype=np.float64)
    b_h = np.asarray(b_h, dtype=np.float64)

    N, T, _ = X.shape
    H = h_0.shape[1]

    h = h_0.copy()

    hidden_states = np.empty((N, T, H), dtype=np.float64)

    for t in range(T):
        x_t = X[:, t, :]                       
        h = np.tanh(x_t @ W_xh.T + h @ W_hh.T + b_h)
        hidden_states[:, t, :] = h            


    return {
        "hidden_states": hidden_states,       
        "final_hidden_state": h.copy(),        
    }