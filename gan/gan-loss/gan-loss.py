import numpy as np

def gan_losses(real_probs: np.ndarray, fake_probs: np.ndarray) -> dict:
    """
    Returns discriminator_loss and generator_loss as Python floats.
    """
    epsilon = 1e-8
    real_probs = np.clip(np.asarray(real_probs, dtype=np.float64), epsilon, 1.0 - epsilon)
    fake_probs = np.clip(np.asarray(fake_probs, dtype=np.float64), epsilon, 1.0 - epsilon)
    
    l_d = -np.mean(np.log(real_probs) + np.log(1.0 - fake_probs))
    l_g = -np.mean(np.log(fake_probs))

    return {"discriminator_loss": l_d, "generator_loss": l_g}