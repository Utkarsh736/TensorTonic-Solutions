import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Returns the float64 pooled tensor in NHWC layout.
    """
    B, H, W, C = x.shape
    reshaped = x.reshape(B, H//2, 2, W//2, 2, C)
    pooled = reshaped.max(axis=(2,4))

    return pooled