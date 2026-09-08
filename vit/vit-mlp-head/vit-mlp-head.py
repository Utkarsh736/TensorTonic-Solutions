import numpy as np

def classification_head(encoder_output: np.ndarray,
                        W_head: np.ndarray) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C).
    """
    cls_tkn = encoder_output[:,0,:]

    mean = np.mean(cls_tkn, axis=-1, keepdims=True)
    var = np.var(cls_tkn, axis=-1, keepdims=True)
    eps = 1e-6

    norm = (cls_tkn-mean)/np.sqrt(var+eps)

    logits = np.matmul(norm, W_head)

    return logits
    