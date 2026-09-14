import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    param_arr = np.asarray(param, dtype=float)
    grad_arr = np.asarray(grad, dtype=float)
    m_arr = np.asarray(m, dtype=float)
    v_arr = np.asarray(v, dtype=float)

    m_new = beta1*m_arr + (1-beta1)*grad_arr
    v_new = beta2*v_arr + (1-beta2)*(grad_arr**2)

    m_hat = m_new/(1-(beta1**t))
    v_hat = v_new/(1-(beta2**t))

    param_new = param_arr - lr * m_hat/(np.sqrt(v_hat)+eps)
    
    return param_new, m_new, v_new
    