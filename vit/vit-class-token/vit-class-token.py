import numpy as np

def prepend_class_token(patches: np.ndarray,
                        cls_token: np.ndarray) -> np.ndarray:
    """
    Returns the float64 sequence with the class token at position zero.
    """
    batch_size = patches.shape[0]

    cls_broadcasted = np.broadcast_to(cls_token, (batch_size, 1, cls_token.shape[2]))

    result = np.concatenate([cls_broadcasted, patches], axis=1)

    return result.astype(np.float64)