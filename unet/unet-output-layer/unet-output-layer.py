import numpy as np

def unet_output(features: np.ndarray, W_out: np.ndarray,
                b_out: np.ndarray) -> np.ndarray:
    """
    Returns float64 per-pixel class logits in NHWC layout.
    """
    return features@W_out + b_out