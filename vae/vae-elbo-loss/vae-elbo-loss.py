import numpy as np

def vae_loss(x: np.ndarray, reconstruction: np.ndarray,
             mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Returns total_loss, reconstruction_loss, and kl_loss as Python floats.
    """
    l_recon = np.mean(np.sum((x-reconstruction)**2, axis=1))
    l_kl = np.mean(np.sum(-0.5*(1 + log_var - mu**2 - np.exp(log_var)), axis=1))
    l_total = float(l_recon + l_kl)

    return {"total_loss": l_total, "reconstruction_loss": l_recon, "kl_loss": l_kl}
                      