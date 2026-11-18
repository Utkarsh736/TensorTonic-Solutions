import numpy as np

def mix_root_dirichlet_noise(
    priors: np.ndarray,
    noise: np.ndarray,
    legal_mask: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """
    Returns the mixed float64 prior array in the original action order.
    """
    priors_f = np.asarray(priors, dtype=np.float64).copy()
    noise_f = np.asarray(noise, dtype=np.float64).copy()
    legal = np.asarray(legal_mask, dtype=bool).copy()

    prior_mass = np.sum(priors_f[legal])
    noise_mass = np.sum(noise_f[legal])

    prior_norm = priors_f[legal] / prior_mass
    noise_norm = noise_f[legal] / noise_mass

    mixed_legal = (1.0 - epsilon) * prior_norm + epsilon * noise_norm

    mixed = np.zeros_like(priors_f, dtype=np.float64)
    mixed[legal] = mixed_legal

    return mixed
