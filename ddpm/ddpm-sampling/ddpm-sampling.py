import numpy as np

def ddpm_sample(x_T: list, betas: list[float], epsilon_preds: list, z_values: list) -> list:
    """
    Returns the final denoised sample rounded to four decimals.
    """
    betas = np.asarray(betas, dtype=np.float64)
    x = np.asarray(x_T, dtype=np.float64)
    epsilon_preds = np.asarray(epsilon_preds, dtype=np.float64)
    z_values = np.asarray(z_values, dtype=np.float64)

    T = len(betas)

    alphas = 1-betas
    alpha_bar = np.cumprod(alphas)

    for i, t in enumerate(range(T, 0, -1)):
        beta_t = betas[t-1]
        alpha_t = alphas[t-1]
        alpha_bar_t = alpha_bar[t-1]

        eps = epsilon_preds[i]

        mu = (x - beta_t*eps/np.sqrt(1-alpha_bar_t))/np.sqrt(alpha_t)

        if t>1: mu = mu + np.sqrt(beta_t) * z_values[i]
        x = mu

    return np.round(x, 4).tolist()