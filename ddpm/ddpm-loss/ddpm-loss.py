import numpy as np

def compute_ddpm_loss(epsilon: list, epsilon_pred: list) -> float:
    """
    Returns the mean DDPM noise-prediction loss.
    """
    epsilon = np.asarray(epsilon, dtype=np.float64)
    epsilon_pred = np.asarray(epsilon_pred, dtype=np.float64)

    diff = epsilon - epsilon_pred

    loss = np.mean(np.square(diff))

    return np.round(loss, 4).item()