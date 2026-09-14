import numpy as np

def adamw_step(w: list, m: list, v: list, grad: list, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w, new_m, and new_v.
    """
    w_arr = np.asarray(w, dtype=float)
    m_arr = np.asarray(m, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    grad_arr = np.asarray(grad, dtype=float)

    new_m = beta1*m_arr + (1-beta1)*grad_arr
    new_v = beta2*v_arr + (1-beta2)*(grad_arr**2)

    adap_stp = lr*new_m/(np.sqrt(new_v)+eps)
    decay_stp = lr * weight_decay * w_arr

    new_w = w_arr - adap_stp - decay_stp

    return {
        "new_w": new_w,
        "new_m": new_m,
        "new_v": new_v,
    }