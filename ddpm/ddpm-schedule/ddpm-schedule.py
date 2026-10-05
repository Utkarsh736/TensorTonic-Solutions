import numpy as np

def linear_beta_schedule(T: int,
                         beta_1: float = 0.0001,
                         beta_T: float = 0.02) -> list[float]:
    """
    Returns T linearly spaced beta values.
    """
    if T==1: return [beta_1]

    noise = np.linspace(beta_1, beta_T, T)

    return np.round(noise, 6).tolist()