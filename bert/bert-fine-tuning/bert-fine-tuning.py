import numpy as np

def bert_fine_tuning_step(hidden_states: np.ndarray, labels: np.ndarray,
                          classifier_W: np.ndarray, classifier_b: np.ndarray,
                          learning_rate: float) -> dict:
    """
    Returns updated classifier parameters and the pre-update loss.
    """
    B, S, D = hidden_states.shape
    h_cls = hidden_states[:, 0, :]
    z = h_cls@classifier_W + classifier_b

    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    p = exp_z/np.sum(exp_z, axis=1, keepdims=True)

    log_probs = np.log(p[np.arange(B), labels])
    loss = -np.mean(log_probs)

    grad_logits = p.copy()
    grad_logits[np.arange(B), labels] -= 1
    grad_logits /= B

    grad_W = h_cls.T @ grad_logits
    grad_b = np.sum(grad_logits, axis=0)

    new_W = classifier_W - learning_rate * grad_W
    new_b = classifier_b - learning_rate * grad_b
    
    return {
        "new_classifier_W": new_W,
        "new_classifier_b": new_b,
        "loss": float(loss)
    }