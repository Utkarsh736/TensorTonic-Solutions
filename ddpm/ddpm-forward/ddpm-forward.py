import numpy as np

def get_alpha_bar(betas: list[float]) -> list[float]:
    """
    Returns the cumulative alpha-bar values rounded to six decimals.
    """
    betas = np.asarray(betas, dtype=np.float64)
    return np.round(np.cumprod(1-betas), 6).tolist()

def forward_diffusion(x_0: list, t: int, betas: list[float], epsilon: list) -> list:
    """
    Returns x_t with the same nested shape as x_0.
    """
    alpha_bar = get_alpha_bar(betas)
    alpha = alpha_bar[t-1]
    
    x_0 = np.asarray(x_0, dtype=np.float64)
    epsilon = np.asarray(epsilon, dtype=np.float64)
    
    x_t = np.sqrt(alpha)*x_0 + np.sqrt(1-alpha)*epsilon

    return np.round(x_t, 4).tolist()