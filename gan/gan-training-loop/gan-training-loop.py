import numpy as np

def train_discriminator_step(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    D_W: np.ndarray,
    learning_rate: float,
) -> dict:
    """
    Returns updated discriminator weights and the pre-update loss.
    """
    batch_size = real_data.shape[0]
    logit_real = np.matmul(real_data, D_W)
    logit_fake = np.matmul(fake_data, D_W)

    real_probs = 1/(1 + np.exp(-logit_real))
    fake_probs = 1/(1 + np.exp(-logit_fake))

    epsilon = 1e-8
    clipped_real = np.clip(real_probs, epsilon, 1-epsilon)
    clipped_fake = np.clip(fake_probs, epsilon, 1-epsilon)

    l_d = -np.mean(np.log(clipped_real) + np.log(1-clipped_fake))
    grad = (np.matmul(real_data.T, real_probs-1) + np.matmul(fake_data.T, fake_probs))/batch_size

    return {"new_discriminator_weights": D_W - learning_rate*grad, "discriminator_loss": l_d}

    