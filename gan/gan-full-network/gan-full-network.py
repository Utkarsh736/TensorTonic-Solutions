import numpy as np

def gan_forward(
    z: np.ndarray,
    real_data: np.ndarray,
    G_W: np.ndarray,
    G_b: np.ndarray,
    D_W: np.ndarray,
) -> dict:
    """
    Returns generated samples, probabilities, and both GAN losses.
    """

    x = np.tanh(np.matmul(z, G_W) + G_b)

    real_logits = np.matmul(real_data, D_W)
    fake_logits = np.matmul(x, D_W)

    real_probs = 1/(1 + np.exp(-real_logits))
    fake_probs = 1/(1 + np.exp(-fake_logits))

    epsilon = 1e-8
    real_clip = np.clip(real_probs, epsilon, 1-epsilon)
    fake_clip = np.clip(fake_probs, epsilon, 1-epsilon)

    l_d = -np.mean(np.log(real_clip) + np.log(1-fake_clip))
    l_g = -np.mean(np.log(fake_clip))

    return {
        "generated_samples": x,
        "real_probabilities": real_probs,
        "fake_probabilities": fake_probs,
        "discriminator_loss": l_d,
        "generator_loss": l_g,
    }